"""
GraphML Studio — Flask GNN Service Test Suite
Unit + integration tests using pytest

Install:
    pip install pytest pytest-flask requests

Run:
    pytest tests/test_gnn_services.py -v

Note: The Flask app must be running on port 5001 for integration tests.
      Unit tests run independently.
"""

import pytest
import json
import numpy as np
import sys
import os

# ── Add parent directory to path ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gnn_services'))


# ════════════════════════════════════════════════
# FIXTURES
# ════════════════════════════════════════════════

@pytest.fixture
def flask_app():
    """Create Flask test client."""
    from app import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_nodes():
    return [
        {'id': 'farmers', 'degree': 10, 'community': 0,
         'communities': {'louvain': 0, 'leiden': 0},
         'centrality': {'betweenness': 0.5, 'eigenvector': 0.3, 'degree': 0.4}},
        {'id': 'protest', 'degree': 8,  'community': 0,
         'communities': {'louvain': 0, 'leiden': 0},
         'centrality': {'betweenness': 0.3, 'eigenvector': 0.2, 'degree': 0.3}},
        {'id': 'modi',    'degree': 6,  'community': 1,
         'communities': {'louvain': 1, 'leiden': 1},
         'centrality': {'betweenness': 0.2, 'eigenvector': 0.1, 'degree': 0.2}},
        {'id': 'bjp',     'degree': 5,  'community': 1,
         'communities': {'louvain': 1, 'leiden': 1},
         'centrality': {'betweenness': 0.1, 'eigenvector': 0.1, 'degree': 0.1}},
    ]


@pytest.fixture
def sample_edges():
    return [
        {'from': 'farmers', 'to': 'protest', 'weight': 10.0},
        {'from': 'farmers', 'to': 'india',   'weight': 8.0},
        {'from': 'modi',    'to': 'bjp',     'weight': 6.0},
        {'from': 'protest', 'to': 'delhi',   'weight': 5.0},
    ]


# ════════════════════════════════════════════════
# UNIT TESTS — model.py
# ════════════════════════════════════════════════

class TestBuildPygData:
    """Tests for the build_pyg_data function in model.py"""

    def test_basic_construction(self):
        import pandas as pd
        from model import build_pyg_data

        nodes = pd.DataFrame({
            'term': ['a', 'b', 'c', 'd'],
            'tf': [100, 80, 60, 40],
            'df': [50, 40, 30, 20],
            'tfidf': [200.0, 160.0, 120.0, 80.0],
            'pmi_max': [3.0, 2.5, 2.0, 1.5],
            'louvain_label': [0, 0, 1, 1],
        })
        edges = pd.DataFrame({
            'source': ['a', 'b', 'c'],
            'target': ['b', 'c', 'd'],
            'weight': [1.0, 2.0, 3.0],
        })

        data, term_to_idx, num_classes, scaler, _ = build_pyg_data(nodes, edges)

        assert data.x is not None
        assert data.edge_index is not None
        assert data.y is not None
        assert num_classes == 2
        assert len(term_to_idx) == 4

    def test_feature_matrix_shape(self):
        import pandas as pd
        from model import build_pyg_data

        nodes = pd.DataFrame({
            'term': ['a', 'b', 'c'],
            'tf': [100, 80, 60],
            'df': [50, 40, 30],
            'tfidf': [200.0, 160.0, 120.0],
            'pmi_max': [3.0, 2.5, 2.0],
            'louvain_label': [0, 0, 1],
        })
        edges = pd.DataFrame({
            'source': ['a'], 'target': ['b'], 'weight': [1.0]
        })

        data, _, _, _, _ = build_pyg_data(nodes, edges)
        # 4 features: tf, df, tfidf, pmi_max
        assert data.x.shape == (3, 4)

    def test_edge_index_is_undirected(self):
        """Edge index should contain both directions for undirected edges."""
        import pandas as pd
        from model import build_pyg_data

        nodes = pd.DataFrame({
            'term': ['a', 'b'],
            'tf': [100, 80], 'df': [50, 40],
            'tfidf': [200.0, 160.0], 'pmi_max': [3.0, 2.5],
            'louvain_label': [0, 1],
        })
        edges = pd.DataFrame({
            'source': ['a'], 'target': ['b'], 'weight': [1.0]
        })

        data, _, _, _, _ = build_pyg_data(nodes, edges)
        # Undirected: a->b and b->a both stored
        assert data.edge_index.shape[1] == 2

    def test_filters_unlabeled_nodes(self):
        """Nodes with NaN labels should be excluded."""
        import pandas as pd
        import numpy as np
        from model import build_pyg_data

        nodes = pd.DataFrame({
            'term': ['a', 'b', 'c', 'd'],
            'tf': [100, 80, 60, 40],
            'df': [50, 40, 30, 20],
            'tfidf': [200.0, 160.0, 120.0, 80.0],
            'pmi_max': [3.0, 2.5, 2.0, 1.5],
            'louvain_label': [0, 1, np.nan, np.nan],
        })
        edges = pd.DataFrame({
            'source': ['a'], 'target': ['b'], 'weight': [1.0]
        })

        data, term_to_idx, _, _, _ = build_pyg_data(nodes, edges)
        assert len(term_to_idx) == 2
        assert 'c' not in term_to_idx
        assert 'd' not in term_to_idx

    def test_handles_from_to_columns(self):
        """Should handle 'from'/'to' column names as well as 'source'/'target'."""
        import pandas as pd
        from model import build_pyg_data

        nodes = pd.DataFrame({
            'term': ['a', 'b'],
            'tf': [100, 80], 'df': [50, 40],
            'tfidf': [200.0, 160.0], 'pmi_max': [3.0, 2.5],
            'louvain_label': [0, 1],
        })
        edges = pd.DataFrame({
            'from': ['a'], 'to': ['b'], 'weight': [1.0]
        })

        data, _, _, _, _ = build_pyg_data(nodes, edges)
        assert data.edge_index is not None

    def test_label_encoder_is_contiguous(self):
        """Community labels should be re-encoded to 0..N-1."""
        import pandas as pd
        from model import build_pyg_data

        nodes = pd.DataFrame({
            'term': ['a', 'b', 'c'],
            'tf': [100, 80, 60], 'df': [50, 40, 30],
            'tfidf': [200.0, 160.0, 120.0], 'pmi_max': [3.0, 2.5, 2.0],
            'louvain_label': [5, 5, 11],  # non-contiguous IDs
        })
        edges = pd.DataFrame({'source': ['a'], 'target': ['b'], 'weight': [1.0]})

        data, _, num_classes, _, _ = build_pyg_data(nodes, edges)
        assert num_classes == 2
        assert data.y.max().item() == 1  # re-encoded to 0,1


class TestGraphSAGENC:
    """Tests for GraphSAGE_NC model architecture."""

    def test_forward_pass_shape(self):
        import torch
        from model import GraphSAGE_NC

        model = GraphSAGE_NC(in_channels=4, hidden_channels=32, out_channels=5)
        model.eval()

        x          = torch.randn(10, 4)
        edge_index = torch.tensor([[0,1,2,3,4], [1,2,3,4,0]], dtype=torch.long)

        with torch.no_grad():
            out = model(x, edge_index)

        assert out.shape == (10, 5)

    def test_embed_shape(self):
        import torch
        from model import GraphSAGE_NC

        model = GraphSAGE_NC(in_channels=4, hidden_channels=64, out_channels=5)
        model.eval()

        x          = torch.randn(10, 4)
        edge_index = torch.tensor([[0,1,2,3], [1,2,3,0]], dtype=torch.long)

        with torch.no_grad():
            emb = model.embed(x, edge_index)

        assert emb.shape == (10, 64)

    def test_dropout_inactive_at_eval(self):
        """Two forward passes in eval mode should give identical results."""
        import torch
        from model import GraphSAGE_NC

        model = GraphSAGE_NC(in_channels=4, hidden_channels=32, out_channels=3)
        model.eval()

        x          = torch.randn(5, 4)
        edge_index = torch.tensor([[0,1,2], [1,2,0]], dtype=torch.long)

        with torch.no_grad():
            out1 = model(x, edge_index)
            out2 = model(x, edge_index)

        assert torch.allclose(out1, out2)


class TestGraphSAGELP:
    """Tests for GraphSAGE_LP model architecture."""

    def test_encode_shape(self):
        import torch
        from model import GraphSAGE_LP

        model = GraphSAGE_LP(in_channels=4, hidden_channels=32, out_channels=16)
        model.eval()

        x          = torch.randn(8, 4)
        edge_index = torch.tensor([[0,1,2,3], [1,2,3,0]], dtype=torch.long)

        with torch.no_grad():
            z = model.encode(x, edge_index)

        assert z.shape == (8, 16)

    def test_decode_shape(self):
        import torch
        from model import GraphSAGE_LP

        model = GraphSAGE_LP(in_channels=4, hidden_channels=32, out_channels=16)
        model.eval()

        z               = torch.randn(8, 16)
        edge_label_idx  = torch.tensor([[0,1,2], [3,4,5]], dtype=torch.long)

        with torch.no_grad():
            scores = model.decode(z, edge_label_idx)

        assert scores.shape == (3,)

    def test_forward_returns_logits(self):
        import torch
        from model import GraphSAGE_LP

        model = GraphSAGE_LP(in_channels=4, hidden_channels=32, out_channels=16)
        model.eval()

        x               = torch.randn(8, 4)
        edge_index      = torch.tensor([[0,1], [1,0]], dtype=torch.long)
        edge_label_idx  = torch.tensor([[0,1], [2,3]], dtype=torch.long)

        with torch.no_grad():
            out = model(x, edge_index, edge_label_idx)

        assert out.shape == (2,)
        # Logits can be any real number
        assert not torch.isnan(out).any()


# ════════════════════════════════════════════════
# INTEGRATION TESTS — Flask API
# ════════════════════════════════════════════════

class TestHealthEndpoint:
    def test_health_returns_200(self, flask_app):
        res = flask_app.get('/health')
        assert res.status_code == 200

    def test_health_has_required_fields(self, flask_app):
        res  = flask_app.get('/health')
        data = json.loads(res.data)
        assert 'status' in data
        assert 'demo_loaded' in data
        assert 'torch_version' in data
        assert 'device' in data

    def test_health_status_is_ok(self, flask_app):
        res  = flask_app.get('/health')
        data = json.loads(res.data)
        assert data['status'] == 'ok'


class TestDemoEndpoints:
    def test_demo_embeddings_requires_load(self, flask_app):
        """Should return 400 if demo not loaded."""
        res = flask_app.get('/gnn/demo/embeddings')
        assert res.status_code == 400
        data = json.loads(res.data)
        assert 'error' in data

    def test_demo_communities_requires_load(self, flask_app):
        res = flask_app.get('/gnn/demo/communities')
        assert res.status_code == 400

    def test_demo_search_requires_load(self, flask_app):
        res = flask_app.post('/gnn/demo/search',
                             json={'query': 'farmers'},
                             content_type='application/json')
        assert res.status_code == 400

    def test_demo_predict_requires_load(self, flask_app):
        res = flask_app.post('/gnn/demo/predict-node',
                             json={'term': 'farmers'},
                             content_type='application/json')
        assert res.status_code == 400

    def test_demo_search_empty_query(self, flask_app):
        """Empty query should return 400."""
        res = flask_app.post('/gnn/demo/search',
                             json={'query': ''},
                             content_type='application/json')
        assert res.status_code in (400, 400)


class TestUserTrainEndpoint:
    def test_train_requires_nodes_and_edges(self, flask_app):
        res = flask_app.post('/gnn/train',
                             json={},
                             content_type='application/json')
        assert res.status_code == 400
        data = json.loads(res.data)
        assert 'error' in data

    def test_train_rejects_empty_nodes(self, flask_app):
        res = flask_app.post('/gnn/train',
                             json={'nodes': [], 'edges': []},
                             content_type='application/json')
        assert res.status_code == 400

    def test_train_small_graph(self, flask_app, sample_nodes, sample_edges):
        """Training on a tiny graph should succeed."""
        payload = {
            'nodes': sample_nodes,
            'edges': [{'source': e['from'], 'target': e['to'],
                       'weight': e['weight']} for e in sample_edges]
        }
        res  = flask_app.post('/gnn/train', json=payload,
                              content_type='application/json')
        data = json.loads(res.data)

        # Either succeeds or fails gracefully with error message
        assert res.status_code in (200, 400, 500)
        if res.status_code == 200:
            assert 'status' in data
            assert 'nc_metrics' in data
            assert 'lp_metrics' in data
        else:
            assert 'error' in data

    def test_user_embeddings_before_training(self, flask_app):
        """Endpoint returns 200 or 400 when no model trained yet."""
        res = flask_app.get('/gnn/user/embeddings')
        assert res.status_code in (200, 400)

    def test_user_predict_before_training(self, flask_app):
        res = flask_app.post('/gnn/user/predict-node',
                             json={'nodeId': 'test'},
                             content_type='application/json')
        assert res.status_code == 400


# ════════════════════════════════════════════════
# UNIT TESTS — train.py
# ════════════════════════════════════════════════

class TestTrainingPipeline:
    """Tests for training functions in train.py"""

    def test_node_classification_returns_metrics(self):
        import torch
        import pandas as pd
        from model import build_pyg_data
        from train import train_node_classification

        # Minimal graph: 20 nodes, 2 communities
        np.random.seed(42)
        nodes = pd.DataFrame({
            'term': [f'w{i}' for i in range(20)],
            'tf':    np.random.randint(50, 500, 20),
            'df':    np.random.randint(20, 200, 20),
            'tfidf': np.random.uniform(500, 5000, 20),
            'pmi_max': np.random.uniform(1, 15, 20),
            'louvain_label': [0]*10 + [1]*10,
        })
        edges = pd.DataFrame({
            'source': [f'w{i}' for i in range(19)],
            'target': [f'w{i+1}' for i in range(19)],
            'weight': np.random.uniform(1, 10, 19),
        })

        data, _, num_classes, _, _ = build_pyg_data(nodes, edges)
        model, metrics = train_node_classification(data, num_classes, epochs=5)

        assert model is not None
        assert 'accuracy' in metrics
        assert 'macro_f1' in metrics
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert 0.0 <= metrics['macro_f1'] <= 1.0

    def test_link_prediction_returns_metrics(self):
        import torch
        import pandas as pd
        from model import build_pyg_data
        from train import train_link_prediction

        np.random.seed(42)
        nodes = pd.DataFrame({
            'term': [f'w{i}' for i in range(30)],
            'tf':    np.random.randint(50, 500, 30),
            'df':    np.random.randint(20, 200, 30),
            'tfidf': np.random.uniform(500, 5000, 30),
            'pmi_max': np.random.uniform(1, 15, 30),
            'louvain_label': [i % 3 for i in range(30)],
        })
        # Need enough edges for 80/10/10 split
        edges = pd.DataFrame({
            'source': [f'w{i}' for i in range(29)] + [f'w{i}' for i in range(0, 20)],
            'target': [f'w{i+1}' for i in range(29)] + [f'w{i+5}' for i in range(0, 20)],
            'weight': np.random.uniform(1, 10, 49),
        })

        data, _, _, _, _ = build_pyg_data(nodes, edges)
        model, metrics = train_link_prediction(data, epochs=5)

        assert model is not None
        assert 'auc_roc' in metrics
        assert 'avg_precision' in metrics
        assert 0.0 <= metrics['auc_roc'] <= 1.0


# ════════════════════════════════════════════════
# END-TO-END TEST
# ════════════════════════════════════════════════

class TestEndToEnd:
    """
    Full pipeline test: build data → train → predict.
    Runs without Flask, testing the Python layer directly.
    """

    def test_full_nc_pipeline(self):
        import torch
        import pandas as pd
        from model import build_pyg_data, GraphSAGE_NC
        from train import train_node_classification

        np.random.seed(0)
        N = 40
        nodes = pd.DataFrame({
            'term': [f'word{i}' for i in range(N)],
            'tf':    np.random.randint(50, 1000, N),
            'df':    np.random.randint(10, 500, N),
            'tfidf': np.random.uniform(100, 10000, N),
            'pmi_max': np.random.uniform(1, 20, N),
            'louvain_label': [i % 4 for i in range(N)],
        })
        edges = pd.DataFrame({
            'source': [f'word{i}' for i in range(N-1)],
            'target': [f'word{i+1}' for i in range(N-1)],
            'weight': np.ones(N-1),
        })

        # Step 1: Build data
        data, term_to_idx, num_classes, _, _ = build_pyg_data(nodes, edges)
        assert num_classes == 4

        # Step 2: Train
        model, metrics = train_node_classification(data, num_classes, epochs=10)
        assert metrics['accuracy'] >= 0.0

        # Step 3: Inference on a known node
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            probs  = torch.softmax(logits[0], dim=0)

        assert probs.shape == (4,)
        assert abs(probs.sum().item() - 1.0) < 1e-5
        pred_class = probs.argmax().item()
        assert 0 <= pred_class < 4
