#!/usr/bin/env python
"""
TF-IDF + PMI vocabulary selection and PMI-weighted co-occurrence network
built from a large tweets.csv using a sliding window.

Designed to run on Kaggle CPU in a streaming, two-pass fashion.
"""

import argparse
import csv
import html
import json
import math
import os
import re
from collections import Counter, defaultdict

import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
TOKEN_RE = re.compile(r"[a-z]{3,}")
MULTISPACE_RE = re.compile(r"\s+")


def build_stopwords(extra_words):
    stop = set(ENGLISH_STOP_WORDS)
    stop.update({"rt", "amp", "via", "th", "nt"})
    if extra_words:
        stop.update(extra_words)
    return stop


def clean_and_tokenize(text, stop_words):
    if not isinstance(text, str):
        return []
    text = html.unescape(text)
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", " ")
    text = NON_ALPHA_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    if not text:
        return []
    tokens = TOKEN_RE.findall(text)
    if not tokens:
        return []
    return [t for t in tokens if t not in stop_words]


def iter_tokenized_docs(path, text_col, chunksize, max_docs, stop_words):
    read_kwargs = {
        "usecols": [text_col],
        "chunksize": chunksize,
        "dtype": {text_col: "string"},
        "on_bad_lines": "skip",
    }
    if path.endswith(".gz"):
        read_kwargs["compression"] = "gzip"

    seen = 0
    for chunk in pd.read_csv(path, **read_kwargs):
        series = chunk[text_col].fillna("")
        for text in series:
            tokens = clean_and_tokenize(text, stop_words)
            if tokens:
                yield tokens
                seen += 1
                if max_docs and seen >= max_docs:
                    return


def first_pass_counts(path, text_col, chunksize, max_docs, stop_words):
    tf = Counter()
    df = Counter()
    doc_count = 0
    total_tokens = 0

    for tokens in tqdm(
        iter_tokenized_docs(path, text_col, chunksize, max_docs, stop_words),
        desc="Pass 1: TF/DF",
    ):
        doc_count += 1
        total_tokens += len(tokens)
        tf.update(tokens)
        df.update(set(tokens))

    return tf, df, doc_count, total_tokens


def select_vocab(tf, df, doc_count, min_term_freq, min_doc_freq, max_vocab):
    candidates = [
        term
        for term, freq in tf.items()
        if freq >= min_term_freq and df.get(term, 0) >= min_doc_freq
    ]

    scores = []
    for term in candidates:
        idf = math.log((1.0 + doc_count) / (1.0 + df[term])) + 1.0
        score = tf[term] * idf
        scores.append((term, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:max_vocab]
    vocab = [t for t, _ in top]
    tfidf_scores = {t: s for t, s in top}
    return vocab, tfidf_scores


def second_pass_cooccurrence(
    path,
    text_col,
    chunksize,
    max_docs,
    stop_words,
    vocab_set,
    window_size,
):
    coocc = Counter()
    total_pair_count = 0

    for tokens in tqdm(
        iter_tokenized_docs(path, text_col, chunksize, max_docs, stop_words),
        desc="Pass 2: Co-occurrence",
    ):
        n = len(tokens)
        if n < 2:
            continue
        for i in range(n - 1):
            ti = tokens[i]
            if ti not in vocab_set:
                continue
            max_j = min(i + window_size, n - 1)
            for j in range(i + 1, max_j + 1):
                tj = tokens[j]
                if tj not in vocab_set:
                    continue
                if ti == tj:
                    continue
                a, b = (ti, tj) if ti < tj else (tj, ti)
                coocc[(a, b)] += 1
                total_pair_count += 1

    return coocc, total_pair_count


def compute_term_pmi_scores(coocc, tf, total_tokens, total_pair_count, min_coocc):
    term_pmi = defaultdict(float)
    if total_pair_count == 0:
        return term_pmi

    for (a, b), c in coocc.items():
        if c < min_coocc:
            continue
        p_xy = c / total_pair_count
        p_x = tf.get(a, 0) / total_tokens
        p_y = tf.get(b, 0) / total_tokens
        if p_xy <= 0 or p_x <= 0 or p_y <= 0:
            continue
        pmi = math.log(p_xy / (p_x * p_y), 2)
        if pmi <= 0:
            continue
        if pmi > term_pmi[a]:
            term_pmi[a] = pmi
        if pmi > term_pmi[b]:
            term_pmi[b] = pmi

    return term_pmi


def finalize_vocab(vocab, term_pmi, pmi_threshold, min_vocab):
    if pmi_threshold is None:
        return set(vocab)

    filtered = [t for t in vocab if term_pmi.get(t, 0.0) >= pmi_threshold]
    if len(filtered) >= min_vocab:
        return set(filtered)

    # fallback: keep top terms by PMI if we dropped too many
    ranked = sorted(vocab, key=lambda t: term_pmi.get(t, 0.0), reverse=True)
    return set(ranked[: max(min_vocab, 1)])


def build_graph_and_write_edges(
    coocc,
    tf,
    df,
    tfidf_scores,
    term_pmi,
    total_tokens,
    total_pair_count,
    vocab_final,
    min_coocc,
    min_ppmi,
    edges_path,
):
    G = nx.Graph()
    G.add_nodes_from(vocab_final)

    for term in vocab_final:
        G.nodes[term]["tf"] = int(tf.get(term, 0))
        G.nodes[term]["df"] = int(df.get(term, 0))
        G.nodes[term]["tfidf"] = float(tfidf_scores.get(term, 0.0))
        G.nodes[term]["pmi_max"] = float(term_pmi.get(term, 0.0))

    if total_pair_count == 0:
        return G

    os.makedirs(os.path.dirname(edges_path), exist_ok=True)
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "coocc", "ppmi", "weight"])

        for (a, b), c in coocc.items():
            if c < min_coocc:
                continue
            if a not in vocab_final or b not in vocab_final:
                continue
            p_xy = c / total_pair_count
            p_x = tf.get(a, 0) / total_tokens
            p_y = tf.get(b, 0) / total_tokens
            if p_xy <= 0 or p_x <= 0 or p_y <= 0:
                continue
            pmi = math.log(p_xy / (p_x * p_y), 2)
            if pmi <= 0:
                continue
            if pmi < min_ppmi:
                continue
            weight = pmi * c
            writer.writerow([a, b, c, round(pmi, 6), round(weight, 6)])
            G.add_edge(a, b, weight=float(weight), coocc=int(c), ppmi=float(pmi))

    return G


def disparity_filter(G, alpha, mode="or"):
    if G.number_of_edges() == 0:
        return G.copy()

    keep = defaultdict(bool)
    alpha_by_edge = defaultdict(list)

    for u in G.nodes():
        neighbors = list(G[u].items())
        k = len(neighbors)
        if k == 0:
            continue
        if k == 1:
            v, _ = neighbors[0]
            key = (u, v) if u <= v else (v, u)
            if mode == "or":
                keep[key] = True
            else:
                alpha_by_edge[key].append(0.0)
            continue

        s = sum(attr.get("weight", 0.0) for _, attr in neighbors)
        if s <= 0:
            continue

        for v, attr in neighbors:
            p = attr.get("weight", 0.0) / s
            alpha_ij = (1.0 - p) ** (k - 1)
            key = (u, v) if u <= v else (v, u)
            if mode == "or":
                if alpha_ij < alpha:
                    keep[key] = True
            else:
                alpha_by_edge[key].append(alpha_ij)

    if mode != "or":
        for key, values in alpha_by_edge.items():
            if len(values) == 2 and values[0] < alpha and values[1] < alpha:
                keep[key] = True

    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for (u, v), keep_edge in keep.items():
        if not keep_edge:
            continue
        if G.has_edge(u, v):
            H.add_edge(u, v, **G.edges[u, v])
    return H


def prune_to_density(G, target_density, mode="or", max_iter=12):
    if G.number_of_nodes() < 2:
        return G.copy(), 1.0

    base_density = nx.density(G)
    if base_density <= target_density:
        return G.copy(), 0.99

    low, high = 1e-6, 0.99
    best_graph = G.copy()
    best_alpha = high

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        H = disparity_filter(G, mid, mode=mode)
        density = nx.density(H) if H.number_of_nodes() > 1 else 0.0
        if density > target_density:
            high = mid
        else:
            low = mid
            best_graph = H
            best_alpha = mid

    return best_graph, best_alpha


def write_edges_from_graph(G, edges_path):
    os.makedirs(os.path.dirname(edges_path), exist_ok=True)
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "coocc", "ppmi", "weight"])
        for u, v, attr in G.edges(data=True):
            writer.writerow(
                [u, v, attr.get("coocc", 0), attr.get("ppmi", 0.0), attr.get("weight", 0.0)]
            )


def write_nodes(G, nodes_path):
    os.makedirs(os.path.dirname(nodes_path), exist_ok=True)
    with open(nodes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["term"]
        if G.number_of_nodes() > 0:
            sample_attrs = next(iter(G.nodes(data=True)))[1]
            header.extend(sample_attrs.keys())
        writer.writerow(header)
        for term, attrs in G.nodes(data=True):
            row = [term]
            for key in header[1:]:
                row.append(attrs.get(key, ""))
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to tweets.csv")
    parser.add_argument("--text-col", default="renderedContent")
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-term-freq", type=int, default=50)
    parser.add_argument("--min-doc-freq", type=int, default=5)
    parser.add_argument("--max-vocab", type=int, default=15000)
    parser.add_argument("--min-coocc", type=int, default=5)
    parser.add_argument("--min-ppmi", type=float, default=0.0)
    parser.add_argument("--pmi-threshold", type=float, default=1.0)
    parser.add_argument("--min-vocab", type=int, default=5000)
    parser.add_argument("--target-density", type=float, default=0.05)
    parser.add_argument("--disparity-mode", choices=["or", "and"], default="or")
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--extra-stopwords", default="")
    parser.add_argument("--outdir", default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()

    stop_words = build_stopwords(
        [w.strip().lower() for w in args.extra_stopwords.split(",") if w.strip()]
    )

    tf, df, doc_count, total_tokens = first_pass_counts(
        args.input,
        args.text_col,
        args.chunksize,
        args.max_docs,
        stop_words,
    )

    vocab, tfidf_scores = select_vocab(
        tf, df, doc_count, args.min_term_freq, args.min_doc_freq, args.max_vocab
    )
    vocab_set = set(vocab)

    coocc, total_pair_count = second_pass_cooccurrence(
        args.input,
        args.text_col,
        args.chunksize,
        args.max_docs,
        stop_words,
        vocab_set,
        args.window,
    )

    term_pmi = compute_term_pmi_scores(
        coocc, tf, total_tokens, total_pair_count, args.min_coocc
    )

    vocab_final = finalize_vocab(vocab, term_pmi, args.pmi_threshold, args.min_vocab)

    os.makedirs(args.outdir, exist_ok=True)
    edges_raw_path = os.path.join(args.outdir, "edges_pmi.csv")

    G = build_graph_and_write_edges(
        coocc,
        tf,
        df,
        tfidf_scores,
        term_pmi,
        total_tokens,
        total_pair_count,
        vocab_final,
        args.min_coocc,
        args.min_ppmi,
        edges_raw_path,
    )

    pruned_graph, alpha_used = prune_to_density(
        G, args.target_density, mode=args.disparity_mode
    )

    edges_pruned_path = os.path.join(args.outdir, "edges_pmi_pruned.csv")
    nodes_path = os.path.join(args.outdir, "nodes.csv")
    gexf_path = os.path.join(args.outdir, "network_pmi_pruned.gexf")

    write_edges_from_graph(pruned_graph, edges_pruned_path)
    write_nodes(pruned_graph, nodes_path)
    nx.write_gexf(pruned_graph, gexf_path)

    stats = {
        "documents": doc_count,
        "tokens_total": total_tokens,
        "vocab_tfidf": len(vocab),
        "vocab_final": len(vocab_final),
        "edges_raw": G.number_of_edges(),
        "edges_pruned": pruned_graph.number_of_edges(),
        "density_raw": nx.density(G) if G.number_of_nodes() > 1 else 0.0,
        "density_pruned": nx.density(pruned_graph)
        if pruned_graph.number_of_nodes() > 1
        else 0.0,
        "disparity_alpha": alpha_used,
        "window": args.window,
    }

    stats_path = os.path.join(args.outdir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("Done")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
