"""
GraphML Studio — GNN Microservice
Flask server exposing GraphSAGE node classification,
link prediction, and t-SNE embeddings.
Runs on port 5001 alongside the Node.js server on port 3000.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import pandas as pd
import os
import json
import traceback
import hashlib
from datetime import datetime

from model import GraphSAGE_NC, GraphSAGE_LP, build_pyg_data
from train import train_node_classification, train_link_prediction
from sklearn.manifold import TSNE

app = Flask(__name__)
CORS(app)

# ── Paths ──
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR   = os.path.join(BASE_DIR, 'demo_data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

device = torch.device('cpu')  # CPU-only for local deployment

# ── In-memory state ──
state = {
    'demo_loaded':    False,
    'demo_data':      None,   # PyG Data object
    'demo_model_nc':  None,
    'demo_model_lp':  None,
    'demo_embeddings': None,  # np array (N, 2) t-SNE coords
    'demo_nodes_df':  None,
    'demo_meta':      None,   # communities.json content
    'user_data':      None,
    'user_model_nc':  None,
    'user_model_lp':  None,
    'user_embeddings': None,
    'user_nodes_df':  None,
    'user_term_to_idx': None,
    'user_scaler': None,
    'user_feature_cols': None,
    'user_graph_hash': None,
    'user_cached': False,
}

def graph_hash(nodes_list, edges_list):
    """
    Deterministic hash of graph structure + basic labels.
    Uses undirected edge normalization and stable ordering.
    """
    node_keys = []
    for n in nodes_list or []:
        nid = str(n.get('id', ''))
        if not nid:
            continue
        comm = n.get('community', '')
        node_keys.append(f"{nid}|{comm}")
    node_keys.sort()

    edge_keys = []
    for e in edges_list or []:
        s = str(e.get('source', e.get('from', '')))
        t = str(e.get('target', e.get('to', '')))
        if not s or not t:
            continue
        if s > t:
            s, t = t, s
        w = e.get('weight', 1.0)
        try:
            w = float(w)
        except Exception:
            w = 1.0
        edge_keys.append(f"{s}|{t}|{w:.6f}")
    edge_keys.sort()

    h = hashlib.sha256()
    for k in node_keys:
        h.update(k.encode('utf-8'))
        h.update(b'\n')
    h.update(b'--edges--\n')
    for k in edge_keys:
        h.update(k.encode('utf-8'))
        h.update(b'\n')
    return h.hexdigest()[:12]

def cache_paths(graph_hash_id):
    return {
        'nc': os.path.join(MODELS_DIR, f'cache_{graph_hash_id}_nc.pt'),
        'lp': os.path.join(MODELS_DIR, f'cache_{graph_hash_id}_lp.pt'),
        'meta': os.path.join(MODELS_DIR, f'cache_{graph_hash_id}.json'),
    }

def load_cache_meta(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def make_tsne(n_components, perplexity, random_state, n_iter):
    """
    Use max_iter when available (sklearn >=1.5) to avoid FutureWarning,
    fallback to n_iter for older versions.
    """
    try:
        return TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            max_iter=n_iter
        )
    except TypeError:
        return TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=n_iter
        )

def parse_connections(raw_connections):
    """
    Normalize connections input into (ids, weights).
    Supports list[str] or list[{'id':..., 'weight':...}].
    """
    ids = []
    weights = []
    for item in raw_connections or []:
        if isinstance(item, dict):
            nid = str(item.get('id') or item.get('node') or item.get('name') or '').strip()
            w = item.get('weight', 1.0)
        else:
            nid = str(item).strip()
            w = 1.0
        if not nid:
            continue
        try:
            w = float(w)
        except Exception:
            w = 1.0
        ids.append(nid)
        weights.append(w)
    return ids, weights

def build_new_node_feature(feature_cols, scaler, conn_ids, conn_weights):
    """
    Build a single-node feature vector matching training feature_cols,
    using connection-derived statistics (degree, strength, mean_weight, etc.).
    """
    degree = len(conn_ids)
    strength = float(sum(conn_weights)) if conn_weights else 0.0
    mean_weight = (strength / degree) if degree > 0 else 0.0
    total_coocc = strength

    base = {
        'degree': degree,
        'strength': strength,
        'mean_weight': mean_weight,
        'total_coocc': total_coocc,
        'tf': 0.0,
        'df': 0.0,
        'tfidf': 0.0,
        'pmi_max': 0.0,
    }

    vec = []
    for col in feature_cols:
        vec.append(float(base.get(col, 0.0)))
    vec = np.array(vec, dtype=np.float32).reshape(1, -1)

    for i, col in enumerate(feature_cols):
        if col in ('tf', 'df', 'tfidf', 'strength', 'total_coocc'):
            vec[:, i] = np.log1p(vec[:, i])

    vec_scaled = scaler.transform(vec).astype(np.float32)
    return torch.tensor(vec_scaled, dtype=torch.float)

def build_inductive_graph(data, term_to_idx, new_feat, conn_ids):
    """
    Extend graph with a new node connected to known neighbors.
    Returns (x_ext, edge_index_ext, new_idx, neighbor_idxs).
    """
    new_idx = data.num_nodes
    neighbor_idxs = []
    src = []
    dst = []
    for nid in conn_ids:
        if nid not in term_to_idx:
            continue
        ni = term_to_idx[nid]
        neighbor_idxs.append(ni)
        src.extend([new_idx, ni])
        dst.extend([ni, new_idx])

    if not neighbor_idxs:
        return None, None, None, []

    x_ext = torch.cat([data.x, new_feat.to(data.x.device)], dim=0)
    add_edges = torch.tensor([src, dst], dtype=torch.long, device=data.edge_index.device)
    edge_index_ext = torch.cat([data.edge_index, add_edges], dim=1)
    return x_ext, edge_index_ext, new_idx, neighbor_idxs

def get_neighbors(edge_index, node_idx):
    src = edge_index[0]
    dst = edge_index[1]
    mask = (src == node_idx)
    if mask.sum().item() == 0:
        return set()
    return set(dst[mask].tolist())

# ──────────────────────────────────────────────
# ROOT
@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'status': 'ok',
        'service': 'graphml-studio-gnn',
        'endpoints': [
            '/health',
            '/gnn/train',
            '/gnn/user/predict-node',
            '/gnn/user/predict-edge',
            '/gnn/user/embeddings',
            '/gnn/demo/load',
            '/gnn/demo/communities',
            '/gnn/demo/embeddings',
        ],
    })

# HEALTH CHECK
# ──────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'demo_loaded': state['demo_loaded'],
        'torch_version': torch.__version__,
        'device': str(device),
    })


# ──────────────────────────────────────────────
# DEMO MODE — Load pre-trained Farmer's Protest
# ──────────────────────────────────────────────
@app.route('/gnn/demo/load', methods=['POST'])
def demo_load():
    try:
        nodes_path = os.path.join(DEMO_DIR, 'nodes_demo.csv')
        edges_path = os.path.join(DEMO_DIR, 'edges_demo.csv')
        emb_path   = os.path.join(DEMO_DIR, 'node_embeddings.csv')
        meta_path  = os.path.join(DEMO_DIR, 'communities.json')
        nc_path    = os.path.join(DEMO_DIR, 'best_sage_nc_v2.pt')
        lp_path    = os.path.join(DEMO_DIR, 'best_sage_lp.pt')

        for p in [nodes_path, edges_path, emb_path, meta_path, nc_path, lp_path]:
            if not os.path.exists(p):
                return jsonify({'error': f'Demo file missing: {os.path.basename(p)}'}), 400

        # Load nodes + edges
        nodes_df = pd.read_csv(nodes_path)
        edges_df = pd.read_csv(edges_path)
        state['demo_nodes_df'] = nodes_df

        # Load community metadata
        with open(meta_path) as f:
            state['demo_meta'] = json.load(f)

        # Build PyG data
        data, term_to_idx, num_classes, scaler, feature_cols = build_pyg_data(nodes_df, edges_df)
        data = data.to(device)
        state['demo_data'] = data

        # Load NC model
        model_nc = GraphSAGE_NC(
            in_channels=data.num_node_features,
            hidden_channels=128,
            out_channels=num_classes,
        ).to(device)
        model_nc.load_state_dict(torch.load(nc_path, map_location=device))
        model_nc.eval()
        state['demo_model_nc'] = model_nc
        state['term_to_idx']   = term_to_idx
        state['num_classes']   = num_classes

        # Load LP model
        model_lp = GraphSAGE_LP(
            in_channels=data.num_node_features,
            hidden_channels=128,
            out_channels=64,
        ).to(device)
        model_lp.load_state_dict(torch.load(lp_path, map_location=device))
        model_lp.eval()
        state['demo_model_lp'] = model_lp

        # Load pre-computed embeddings + compute t-SNE
        emb_df = pd.read_csv(emb_path)
        emb_cols = [c for c in emb_df.columns if c.startswith('emb_')]
        embeddings = emb_df[emb_cols].values.astype(np.float32)

        print(f'Computing t-SNE on {len(embeddings)} nodes...')
        tsne   = make_tsne(n_components=2, perplexity=40, random_state=42, n_iter=1000)
        emb_2d = tsne.fit_transform(embeddings)
        state['demo_embeddings'] = emb_2d
        state['demo_emb_df']     = emb_df
        state['demo_loaded']     = True

        print(f'Demo loaded: {len(nodes_df)} nodes, {len(edges_df)} edges')
        return jsonify({
            'status':      'ok',
            'nodes':       len(nodes_df),
            'edges':       len(edges_df),
            'communities': num_classes,
            'message':     "Farmer's Protest demo loaded successfully",
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────────────────────
# DEMO — Get t-SNE embeddings for visualisation
# ──────────────────────────────────────────────
@app.route('/gnn/demo/embeddings', methods=['GET'])
def demo_embeddings():
    if not state['demo_loaded']:
        return jsonify({'error': 'Demo not loaded. Call /gnn/demo/load first.'}), 400
    try:
        emb_2d   = state['demo_embeddings']
        emb_df   = state['demo_emb_df']
        meta     = state['demo_meta']
        nodes_df = state['demo_nodes_df']

        points = []
        for i, row in emb_df.iterrows():
            comm_id = int(row['louvain_label'])
            theme   = meta['themes'].get(str(comm_id), f'Community {comm_id}')
            points.append({
                'term':      row['term'],
                'x':         float(emb_2d[i, 0]),
                'y':         float(emb_2d[i, 1]),
                'community': comm_id,
                'theme':     theme,
            })

        return jsonify({
            'points':      points,
            'communities': meta['themes'],
            'colors':      meta.get('colors', {}),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────────────────────
# DEMO — Get community details (top words etc.)
# ──────────────────────────────────────────────
@app.route('/gnn/demo/communities', methods=['GET'])
def demo_communities():
    if not state['demo_loaded']:
        return jsonify({'error': 'Demo not loaded.'}), 400
    try:
        nodes_df = state['demo_nodes_df']
        meta     = state['demo_meta']
        result   = []

        for comm_id_str, theme in meta['themes'].items():
            comm_id   = int(comm_id_str)
            comm_rows = nodes_df[nodes_df['louvain_label'] == comm_id]
            top_words = comm_rows.nlargest(12, 'tfidf')['term'].tolist() \
                        if 'tfidf' in comm_rows.columns else \
                        comm_rows.head(12)['term'].tolist()
            result.append({
                'id':        comm_id,
                'theme':     theme,
                'size':      len(comm_rows),
                'top_words': top_words,
                'color':     meta.get('colors', {}).get(comm_id_str, '#38bdf8'),
            })

        result.sort(key=lambda x: x['size'], reverse=True)
        return jsonify({'communities': result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────────────────────
# DEMO — Node search
# ──────────────────────────────────────────────
@app.route('/gnn/demo/search', methods=['POST'])
def demo_search():
    if not state['demo_loaded']:
        return jsonify({'error': 'Demo not loaded.'}), 400
    try:
        query    = request.json.get('query', '').strip().lower()
        nodes_df = state['demo_nodes_df']
        meta     = state['demo_meta']

        if not query:
            return jsonify({'error': 'Empty query.'}), 400

        # Exact match first, then prefix match
        exact   = nodes_df[nodes_df['term'] == query]
        prefix  = nodes_df[nodes_df['term'].str.startswith(query)].head(8)
        matches = pd.concat([exact, prefix]).drop_duplicates('term').head(8)

        if matches.empty:
            return jsonify({'results': [], 'message': f'No match for "{query}"'})

        results = []
        for _, row in matches.iterrows():
            comm_id = int(row['louvain_label']) if 'louvain_label' in row else -1
            theme   = meta['themes'].get(str(comm_id), f'Community {comm_id}')
            results.append({
                'term':      row['term'],
                'community': comm_id,
                'theme':     theme,
                'tfidf':     float(row.get('tfidf', 0)),
                'pmi_max':   float(row.get('pmi_max', 0)),
                'color':     meta.get('colors', {}).get(str(comm_id), '#38bdf8'),
            })

        return jsonify({'results': results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────────────────────
# DEMO — Node classification inference
# ──────────────────────────────────────────────
@app.route('/gnn/demo/predict-node', methods=['POST'])
def demo_predict_node():
    if not state['demo_loaded']:
        return jsonify({'error': 'Demo not loaded.'}), 400
    try:
        term         = request.json.get('term', '').strip().lower()
        term_to_idx  = state['term_to_idx']
        model_nc     = state['demo_model_nc']
        data         = state['demo_data']
        meta         = state['demo_meta']

        if term not in term_to_idx:
            return jsonify({'error': f'Term "{term}" not in vocabulary.'}), 400

        idx = term_to_idx[term]
        model_nc.eval()
        with torch.no_grad():
            logits = model_nc(data.x, data.edge_index)
            probs  = torch.softmax(logits[idx], dim=0).cpu().numpy()

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class]) * 100
        theme      = meta['themes'].get(str(pred_class), f'Community {pred_class}')

        top5 = sorted(enumerate(probs.tolist()), key=lambda x: x[1], reverse=True)[:5]
        top5_result = [
            {
                'community': c,
                'theme':     meta['themes'].get(str(c), f'Community {c}'),
                'probability': round(p * 100, 2),
            }
            for c, p in top5
        ]

        return jsonify({
            'term':             term,
            'predicted_community': pred_class,
            'theme':            theme,
            'confidence':       round(confidence, 2),
            'top5':             top5_result,
            'color':            meta.get('colors', {}).get(str(pred_class), '#38bdf8'),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────────────────────
# USER UPLOAD — Train GNN on custom graph
# ──────────────────────────────────────────────
@app.route('/gnn/train', methods=['POST'])
def user_train():
    try:
        payload  = request.json or {}
        nodes_list = payload.get('nodes', [])
        edges_list = payload.get('edges', [])
        mode = str(payload.get('mode', 'fast')).lower()
        force = bool(payload.get('force', False))

        if not nodes_list or not edges_list:
            return jsonify({'error': 'nodes and edges required.'}), 400

        # Training hyperparams (fast vs full)
        nc_epochs = int(payload.get('nc_epochs', 20 if mode == 'fast' else 100))
        lp_epochs = int(payload.get('lp_epochs', 20 if mode == 'fast' else 80))
        hidden    = int(payload.get('hidden', 64 if mode == 'fast' else 128))
        lp_out    = int(payload.get('lp_out', 64))

        # Cache key depends on graph + mode + hyperparams
        gh = graph_hash(nodes_list, edges_list)
        cache_id = f"{gh}_{mode}_{hidden}_{nc_epochs}_{lp_epochs}"
        paths = cache_paths(cache_id)
        meta = load_cache_meta(paths['meta'])

        nodes_df = pd.DataFrame(nodes_list)
        edges_df = pd.DataFrame(edges_list)

        # Rename columns to standard format if needed
        if 'from' in edges_df.columns:
            edges_df = edges_df.rename(columns={'from': 'source', 'to': 'target'})

        # Need at least degree as a feature — compute if missing
        if 'degree' not in nodes_df.columns:
            deg_map = {}
            for _, row in edges_df.iterrows():
                deg_map[row['source']] = deg_map.get(row['source'], 0) + 1
                deg_map[row['target']] = deg_map.get(row['target'], 0) + 1
            nodes_df['degree'] = nodes_df['id'].map(deg_map).fillna(0)

        print(f'Training on user graph: {len(nodes_df)} nodes, {len(edges_df)} edges')

        data, term_to_idx, num_classes, scaler, feature_cols = build_pyg_data(
            nodes_df, edges_df, label_col='community', id_col='id'
        )

        if num_classes < 2:
            return jsonify({'error': 'Need at least 2 communities for classification.'}), 400

        data = data.to(device)

        # ── Cache hit: load models and skip training ──
        if (not force and meta and os.path.exists(paths['nc']) and os.path.exists(paths['lp'])):
            model_nc = GraphSAGE_NC(
                in_channels=data.num_node_features,
                hidden_channels=meta.get('hidden', hidden),
                out_channels=num_classes,
            ).to(device)
            model_nc.load_state_dict(torch.load(paths['nc'], map_location=device))
            model_nc.eval()

            model_lp = GraphSAGE_LP(
                in_channels=data.num_node_features,
                hidden_channels=meta.get('hidden', hidden),
                out_channels=meta.get('lp_out', lp_out),
            ).to(device)
            model_lp.load_state_dict(torch.load(paths['lp'], map_location=device))
            model_lp.eval()

            state['user_data']     = data
            state['user_model_nc'] = model_nc
            state['user_model_lp'] = model_lp
            state['user_term_to_idx'] = term_to_idx
            state['user_nodes_df'] = nodes_df
            state['user_scaler'] = scaler
            state['user_feature_cols'] = feature_cols
            state['user_graph_hash'] = cache_id
            state['user_cached'] = True

            with torch.no_grad():
                embeddings = model_nc.embed(data.x, data.edge_index).cpu().numpy()
            tsne   = make_tsne(n_components=2, perplexity=min(40, len(nodes_df)//4),
                               random_state=42, n_iter=500)
            emb_2d = tsne.fit_transform(embeddings)
            state['user_embeddings'] = emb_2d

            return jsonify({
                'status':        'cached',
                'cached':        True,
                'graph_hash':    cache_id,
                'nodes':         len(nodes_df),
                'edges':         len(edges_df),
                'communities':   num_classes,
                'nc_metrics':    meta.get('nc_metrics', {}),
                'lp_metrics':    meta.get('lp_metrics', {}),
                'mode':          meta.get('mode', mode),
            })

        # Train NC
        model_nc, nc_metrics = train_node_classification(
            data, num_classes, epochs=nc_epochs, hidden=hidden
        )
        model_nc.eval()
        state['user_data']     = data
        state['user_model_nc'] = model_nc
        state['user_term_to_idx'] = term_to_idx
        state['user_nodes_df'] = nodes_df
        state['user_scaler'] = scaler
        state['user_feature_cols'] = feature_cols
        state['user_graph_hash'] = cache_id
        state['user_cached'] = False

        # Compute embeddings + t-SNE
        with torch.no_grad():
            embeddings = model_nc.embed(data.x, data.edge_index).cpu().numpy()
        tsne   = make_tsne(n_components=2,
                           perplexity=min(40, len(nodes_df)//4),
                           random_state=42,
                           n_iter=500)
        emb_2d = tsne.fit_transform(embeddings)
        state['user_embeddings'] = emb_2d

        # Train LP
        model_lp, lp_metrics = train_link_prediction(
            data, epochs=lp_epochs, hidden=hidden
        )
        model_lp.eval()
        state['user_model_lp'] = model_lp

        # Save cache
        try:
            torch.save(model_nc.state_dict(), paths['nc'])
            torch.save(model_lp.state_dict(), paths['lp'])
            meta_out = {
                'graph_hash':  cache_id,
                'mode':        mode,
                'hidden':      hidden,
                'lp_out':      lp_out,
                'nc_epochs':   nc_epochs,
                'lp_epochs':   lp_epochs,
                'nodes':       len(nodes_df),
                'edges':       len(edges_df),
                'communities': num_classes,
                'nc_metrics':  nc_metrics,
                'lp_metrics':  lp_metrics,
                'trained_at':  datetime.utcnow().isoformat() + 'Z',
            }
            with open(paths['meta'], 'w') as f:
                json.dump(meta_out, f)
        except Exception:
            pass

        return jsonify({
            'status':        'trained',
            'cached':        False,
            'graph_hash':    cache_id,
            'nodes':         len(nodes_df),
            'edges':         len(edges_df),
            'communities':   num_classes,
            'nc_metrics':    nc_metrics,
            'lp_metrics':    lp_metrics,
            'mode':          mode,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────────────────────
# USER — Get embeddings
# ──────────────────────────────────────────────
@app.route('/gnn/user/embeddings', methods=['GET'])
def user_embeddings():
    if state['user_embeddings'] is None:
        return jsonify({'error': 'No user model trained yet.'}), 400
    try:
        emb_2d   = state['user_embeddings']
        nodes_df = state['user_nodes_df']
        id_col   = 'id' if 'id' in nodes_df.columns else nodes_df.columns[0]

        points = []
        for i, row in nodes_df.iterrows():
            if i >= len(emb_2d):
                break
            points.append({
                'id':        str(row[id_col]),
                'x':         float(emb_2d[i, 0]),
                'y':         float(emb_2d[i, 1]),
                'community': int(row.get('community', 0)),
            })

        return jsonify({'points': points})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────────────────────
# USER — Node prediction
# ──────────────────────────────────────────────
@app.route('/gnn/user/predict-node', methods=['POST'])
def user_predict_node():
    if state['user_model_nc'] is None:
        return jsonify({'error': 'No user model trained yet.'}), 400
    try:
        payload = request.json or {}
        node_id = str(payload.get('nodeId') or payload.get('newNodeId') or '').strip()
        term_to_idx  = state['user_term_to_idx']
        model_nc     = state['user_model_nc']
        data         = state['user_data']
        scaler       = state['user_scaler']
        feature_cols = state['user_feature_cols']

        if not node_id:
            return jsonify({'error': 'nodeId required.'}), 400

        # Case 1: existing node
        if node_id in term_to_idx:
            idx = term_to_idx[node_id]
            model_nc.eval()
            with torch.no_grad():
                logits = model_nc(data.x, data.edge_index)
                probs  = torch.softmax(logits[idx], dim=0).cpu().numpy()

        # Case 2: new node (inductive) — requires known connections
        else:
            conn_raw = payload.get('connections') or payload.get('newConnections') or []
            conn_ids, conn_weights = parse_connections(conn_raw)
            if not conn_ids:
                return jsonify({'error': f'Node "{node_id}" not found. Provide connections to existing nodes.'}), 400
            if scaler is None or feature_cols is None:
                return jsonify({'error': 'Model metadata missing. Retrain the GNN.'}), 400

            new_feat = build_new_node_feature(feature_cols, scaler, conn_ids, conn_weights)
            x_ext, edge_index_ext, idx, _ = build_inductive_graph(
                data, term_to_idx, new_feat, conn_ids
            )
            if idx is None:
                return jsonify({'error': 'No valid connections found in graph.'}), 400

            model_nc.eval()
            with torch.no_grad():
                logits = model_nc(x_ext, edge_index_ext)
                probs  = torch.softmax(logits[idx], dim=0).cpu().numpy()

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class]) * 100

        return jsonify({
            'nodeId':               node_id,
            'predicted_community':  pred_class,
            'confidence':           round(confidence, 2),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ————————————————————————————————————————————————————————
# USER — Link prediction (GraphSAGE)
# ————————————————————————————————————————————————————————
@app.route('/gnn/user/predict-edge', methods=['POST'])
def user_predict_edge():
    if state['user_model_lp'] is None:
        return jsonify({'error': 'No user model trained yet.'}), 400
    try:
        payload = request.json or {}
        node_id = str(payload.get('nodeId') or payload.get('newNodeId') or '').strip()
        top_k   = int(payload.get('topK', 8))

        if not node_id:
            return jsonify({'error': 'nodeId required.'}), 400

        term_to_idx  = state['user_term_to_idx']
        model_lp     = state['user_model_lp']
        data         = state['user_data']
        scaler       = state['user_scaler']
        feature_cols = state['user_feature_cols']

        # Build graph (existing or inductive)
        if node_id in term_to_idx:
            x_use = data.x
            edge_index_use = data.edge_index
            node_idx = term_to_idx[node_id]
            exclude_neighbors = get_neighbors(edge_index_use, node_idx)
            conn_raw = payload.get('connections') or payload.get('newConnections') or []
            conn_ids, _ = parse_connections(conn_raw)
            for nid in conn_ids:
                if nid in term_to_idx:
                    exclude_neighbors.add(term_to_idx[nid])
        else:
            conn_raw = payload.get('connections') or payload.get('newConnections') or []
            conn_ids, conn_weights = parse_connections(conn_raw)
            if not conn_ids:
                return jsonify({'error': f'Node "{node_id}" not found. Provide connections to existing nodes.'}), 400
            if scaler is None or feature_cols is None:
                return jsonify({'error': 'Model metadata missing. Retrain the GNN.'}), 400

            new_feat = build_new_node_feature(feature_cols, scaler, conn_ids, conn_weights)
            x_use, edge_index_use, node_idx, neighbor_idxs = build_inductive_graph(
                data, term_to_idx, new_feat, conn_ids
            )
            if node_idx is None:
                return jsonify({'error': 'No valid connections found in graph.'}), 400
            exclude_neighbors = set(neighbor_idxs)

        # Map idx -> node id
        id_list = [None] * len(term_to_idx)
        for nid, idx in term_to_idx.items():
            if idx < len(id_list):
                id_list[idx] = nid

        model_lp.eval()
        with torch.no_grad():
            z = model_lp.encode(x_use, edge_index_use)

        base_n = data.num_nodes
        if node_idx >= base_n:
            scores = (z[node_idx] * z[:base_n]).sum(dim=1)
            candidate_idxs = range(base_n)
        else:
            scores = (z[node_idx] * z[:base_n]).sum(dim=1)
            candidate_idxs = range(base_n)

        probs = torch.sigmoid(scores).cpu().numpy()
        predictions = []
        for i in candidate_idxs:
            if i == node_idx:
                continue
            if i in exclude_neighbors:
                continue
            nid = id_list[i] if i < len(id_list) else str(i)
            predictions.append({
                'id': str(nid),
                'score': float(scores[i]),
                'prob': round(float(probs[i]) * 100, 2),
            })

        predictions.sort(key=lambda x: x['score'], reverse=True)
        k = max(1, min(20, top_k))
        return jsonify({ 'predictions': predictions[:k] })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print(f'Starting GraphML Studio GNN Service on port {port}...')
    app.run(host='0.0.0.0', port=port, debug=False)
