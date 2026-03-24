"""
Training pipeline for user-uploaded graphs.
Used when a user uploads their own CSV — not used for demo mode
(demo mode loads pre-trained weights directly).
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.transforms import RandomNodeSplit, RandomLinkSplit

from model import GraphSAGE_NC, GraphSAGE_LP

device = torch.device('cpu')


def train_node_classification(data, num_classes, epochs=100, hidden=128,
                               lr=0.005, dropout=0.3, patience=15):
    """
    Train GraphSAGE for node classification on user-uploaded graph.
    Returns trained model + metrics dict.
    """
    # Split nodes
    transform = RandomNodeSplit(split='train_rest', num_val=0.2, num_test=0.2)
    data = transform(data)

    # Class weights for imbalance
    train_labels  = data.y[data.train_mask].cpu()
    class_counts  = torch.bincount(train_labels, minlength=num_classes).float()
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = (class_weights / class_weights.sum() * num_classes).to(device)

    model     = GraphSAGE_NC(data.num_node_features, hidden, num_classes, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8)

    best_val_f1  = 0
    patience_cnt = 0
    best_state   = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask],
                               data.y[data.train_mask],
                               weight=class_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out   = model(data.x, data.edge_index)
            preds = out[data.val_mask].argmax(dim=1).cpu().numpy()
            truth = data.y[data.val_mask].cpu().numpy()
            val_f1 = f1_score(truth, preds, average='macro', zero_division=0)

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            patience_cnt = 0
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    # Test metrics
    model.eval()
    with torch.no_grad():
        out   = model(data.x, data.edge_index)
        preds = out[data.test_mask].argmax(dim=1).cpu().numpy()
        truth = data.y[data.test_mask].cpu().numpy()

    metrics = {
        'accuracy': round(float(accuracy_score(truth, preds)), 4),
        'macro_f1': round(float(f1_score(truth, preds, average='macro',
                                          zero_division=0)), 4),
        'epochs_trained': epoch,
    }
    return model, metrics


def train_link_prediction(data, epochs=80, hidden=128,
                           lr=0.001, dropout=0.3, patience=15):
    """
    Train GraphSAGE for link prediction on user-uploaded graph.
    Returns trained model + metrics dict.
    """
    from sklearn.metrics import average_precision_score

    lp_transform = RandomLinkSplit(
        num_val=0.1, num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
    )
    train_data, val_data, test_data = lp_transform(data)
    train_data = train_data.to(device)
    val_data   = val_data.to(device)
    test_data  = test_data.to(device)

    model     = GraphSAGE_LP(data.num_node_features, hidden, 64, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val_auc = 0
    patience_cnt = 0
    best_state   = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out  = model(train_data.x, train_data.edge_index,
                     train_data.edge_label_index)
        loss = F.binary_cross_entropy_with_logits(out, train_data.edge_label)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z    = model.encode(train_data.x, train_data.edge_index)
            out  = model.decode(z, val_data.edge_label_index)
            prob = torch.sigmoid(out).cpu().numpy()
            true = val_data.edge_label.cpu().numpy()
            val_auc = roc_auc_score(true, prob)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_cnt = 0
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    # Test metrics
    model.eval()
    with torch.no_grad():
        z    = model.encode(train_data.x, train_data.edge_index)
        out  = model.decode(z, test_data.edge_label_index)
        prob = torch.sigmoid(out).cpu().numpy()
        true = test_data.edge_label.cpu().numpy()

    metrics = {
        'auc_roc':       round(float(roc_auc_score(true, prob)), 4),
        'avg_precision': round(float(average_precision_score(true, prob)), 4),
        'epochs_trained': epoch,
    }
    return model, metrics