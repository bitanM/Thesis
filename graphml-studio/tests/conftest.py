"""
conftest.py — pytest configuration
Adds the gnn_services directory to sys.path so that
app.py, model.py and train.py can be imported by the test file.
"""
import sys
import os

# Add gnn_services directory to path
GNN_SERVICE_DIR = os.path.join(
    os.path.dirname(__file__),  # tests/
    '..',                        # graphml-studio/
    'gnn_services'               # gnn_services/
)
sys.path.insert(0, os.path.abspath(GNN_SERVICE_DIR))