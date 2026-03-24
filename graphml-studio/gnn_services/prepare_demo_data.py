"""
prepare_demo_data.py
Run this ONCE after downloading your Kaggle outputs.
Place your downloaded files in the same folder as this script, then run:

    python prepare_demo_data.py

This will create the demo_data/ folder with everything the Flask service needs.

Required input files (download from Kaggle output):
    - node_embeddings.csv
    - best_sage_nc_v2.pt
    - best_sage_lp.pt

Required dataset files (download from Kaggle input dataset):
    - nodes.csv
    - edges_pmi_pruned.csv
"""

import pandas as pd
import numpy as np
import json
import os
import shutil

# ── Output directory ──
DEMO_DIR = os.path.join(os.path.dirname(__file__), 'demo_data')
os.makedirs(DEMO_DIR, exist_ok=True)

# ── Community themes (from our analysis) ──
COMMUNITY_THEMES = {
    "0":  "Counter-Narrative / Fringe",
    "1":  "International Amplification",
    "2":  "Systemic Critique (Adani/Ambani)",
    "3":  "Celebrity Discourse (MIA/Rihanna)",
    "4":  "Farm Law & MSP Policy",
    "5":  "Core Protest Vocabulary",
    "6":  "Agricultural Heritage / Ethos",
    "7":  "Media Criticism & Khalistan Framing",
    "8":  "Cross-Movement Solidarity (CAA/NRC)",
    "9":  "Ground-Level Geography (Delhi/Punjab)",
    "10": "Tech & Climate Crossover",
    "11": "Political Opposition (BJP/RSS)",
    "12": "Viral Mechanics (Hashtag/Retweet)",
    "13": "Punjabi / Sikh Identity",
    "14": "Democratic Grievances & Slogans",
}

# ── Community colors (tab20 palette) ──
COMMUNITY_COLORS = {
    "0":  "#1f77b4", "1":  "#aec7e8", "2":  "#ffbb78", "3":  "#2ca02c",
    "4":  "#98df8a", "5":  "#d62728", "6":  "#ff9896", "7":  "#9467bd",
    "8":  "#c5b0d5", "9":  "#8c564b", "10": "#c49c94", "11": "#e377c2",
    "12": "#f7b6d2", "13": "#7f7f7f", "14": "#bcbd22",
}

print("=" * 55)
print("GraphML Studio — Demo Data Preparation")
print("=" * 55)

# ── Step 1: Copy model weights ──
print("\n[1/5] Copying model weights...")
for fname in ['best_sage_nc_v2.pt', 'best_sage_lp.pt']:
    if os.path.exists(fname):
        shutil.copy(fname, os.path.join(DEMO_DIR, fname))
        print(f"  ✓ {fname}")
    else:
        print(f"  ✗ MISSING: {fname} — download from Kaggle output")

# ── Step 2: Copy node embeddings ──
print("\n[2/5] Processing node_embeddings.csv...")
if os.path.exists('node_embeddings.csv'):
    emb_df = pd.read_csv('node_embeddings.csv')
    emb_df.to_csv(os.path.join(DEMO_DIR, 'node_embeddings.csv'), index=False)
    print(f"  ✓ {len(emb_df)} nodes with embeddings")
else:
    print("  ✗ MISSING: node_embeddings.csv")

# ── Step 3: Process nodes.csv ──
print("\n[3/5] Processing nodes.csv...")
if os.path.exists('nodes.csv'):
    nodes_df = pd.read_csv('nodes.csv')
    # Merge community labels from embeddings if available
    if os.path.exists('node_embeddings.csv'):
        emb_df   = pd.read_csv('node_embeddings.csv')
        label_df = emb_df[['term', 'louvain_label']].copy()
        nodes_df = nodes_df.merge(label_df, on='term', how='left')
    nodes_df.to_csv(os.path.join(DEMO_DIR, 'nodes_demo.csv'), index=False)
    print(f"  ✓ {len(nodes_df)} nodes saved to nodes_demo.csv")
else:
    print("  ✗ MISSING: nodes.csv")

# ── Step 4: Sample edges (keep top 50K by weight for fast loading) ──
print("\n[4/5] Sampling edges_pmi_pruned.csv...")
if os.path.exists('edges_pmi_pruned.csv'):
    edges_df = pd.read_csv('edges_pmi_pruned.csv')
    print(f"  Full edges: {len(edges_df):,}")
    # Keep top 50K edges by weight — enough for meaningful GNN, fast to load
    MAX_EDGES = 50000
    if len(edges_df) > MAX_EDGES:
        edges_sample = edges_df.nlargest(MAX_EDGES, 'weight')
    else:
        edges_sample = edges_df
    edges_sample.to_csv(os.path.join(DEMO_DIR, 'edges_demo.csv'), index=False)
    print(f"  ✓ {len(edges_sample):,} edges saved to edges_demo.csv")
else:
    print("  ✗ MISSING: edges_pmi_pruned.csv")

# ── Step 5: Save communities.json ──
print("\n[5/5] Writing communities.json...")
communities_meta = {
    "themes": COMMUNITY_THEMES,
    "colors": COMMUNITY_COLORS,
    "dataset": "Farmer's Protest Twitter Corpus (2020-2021)",
    "nodes": 14033,
    "edges": 995117,
    "algorithm": "Louvain (NetworkX, weight='weight', seed=42)",
    "nc_metrics": {
        "accuracy": 0.4184,
        "macro_f1": 0.2059,
        "vs_baseline_acc": "+37.1%",
        "vs_baseline_f1":  "+394.9%"
    },
    "lp_metrics": {
        "auc_roc":       0.7697,
        "avg_precision": 0.7301
    }
}
with open(os.path.join(DEMO_DIR, 'communities.json'), 'w') as f:
    json.dump(communities_meta, f, indent=2)
print("  ✓ communities.json saved")

# ── Summary ──
print("\n" + "=" * 55)
print("Demo data folder contents:")
for f in sorted(os.listdir(DEMO_DIR)):
    size = os.path.getsize(os.path.join(DEMO_DIR, f))
    print(f"  {f:<30} {size/1024:.1f} KB")
print("\nDone! You can now start the Flask service:")
print("  cd gnn_services")
print("  python app.py")
