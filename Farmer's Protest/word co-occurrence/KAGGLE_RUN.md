# Kaggle Run Instructions

This repo includes a CPU-friendly, two-pass pipeline that:
- Cleans the tweet corpus (column: renderedContent)
- Selects vocabulary using TF-IDF and PMI
- Builds sliding-window co-occurrence (window size 3-5, default 5)
- Weights edges by PMI-weighted co-occurrence counts
- Prunes edges with the disparity filter to reach density < 0.05

## 1) Upload data to Kaggle
- Create a Kaggle Dataset from your `tweets.csv` (the file is in `Farmer's Protest Data`).
- Add it as input to your Kaggle Notebook.

Assume the dataset appears at:
`/kaggle/input/farmers-protest-data/tweets.csv`

## 2) Add this script to the Kaggle Notebook
Options:
- Copy/paste the contents of `kaggle_tfidf_pmi_pipeline.py` into a Kaggle cell.
- Or upload the script as a dataset and call it with `python`.

## 3) Run
Example command (adjust paths and thresholds as needed):

```bash
python kaggle_tfidf_pmi_pipeline.py \
  --input /kaggle/input/farmers-protest-data/tweets.csv \
  --text-col renderedContent \
  --window 5 \
  --max-vocab 15000 \
  --min-term-freq 50 \
  --min-doc-freq 5 \
  --min-coocc 5 \
  --pmi-threshold 1.0 \
  --target-density 0.05 \
  --outdir /kaggle/working/outputs
```

Outputs are written to `/kaggle/working/outputs`:
- `edges_pmi.csv`
- `edges_pmi_pruned.csv`
- `nodes.csv`
- `network_pmi_pruned.gexf`
- `stats.json`

## 4) Notes / tuning tips
- If memory is tight, reduce `--max-vocab` and increase `--min-term-freq`.
- If the graph is still too dense, increase `--min-ppmi` or lower `--target-density`.
- For faster tests, use `--max-docs 50000`.
