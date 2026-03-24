# GraphML Studio — Test Suite

Two test suites covering both components of the platform.

---

## Structure

```
tests/
├── server.test.js        ← Node.js Jest tests (Express endpoints)
├── test_gnn_services.py   ← Python pytest tests (Flask GNN service)
├── jest.config.json      ← Jest configuration
└── README.md
```

---

## Node.js Tests (Jest + Supertest)

### What is tested
- `detectSeparator()` — CSV delimiter detection (unit)
- `looksLikeHeaderRow()` — header detection (unit)
- `POST /api/analyze` — CSV upload and graph processing (integration)
- `POST /api/predict-edge` — link prediction endpoint (integration)
- `POST /api/predict-node` — node prediction endpoint (integration)
- `GET /api/gnn/status` — GNN service health proxy (integration)
- `GET /` — static file serving (integration)

### Setup

**Step 1 — Export the Express app from server.js**

Add these two lines to the very bottom of `server.js`:
```js
// Replace:
app.listen(3000, () => console.log('...'));

// With:
if (require.main === module) {
  app.listen(3000, () => console.log('✅ GraphML Studio running on http://localhost:3000'));
}
module.exports = app;
```

**Step 2 — Install test dependencies**
```bash
cd graphml-studio
npm install --save-dev jest supertest
```

**Step 3 — Run tests**
```bash
# All tests
npx jest tests/server.test.js --verbose

# Unit tests only (fast, no server needed)
npx jest --testNamePattern="detectSeparator|looksLikeHeaderRow" --verbose

# Integration tests only
npx jest --testNamePattern="POST /api|GET /api|Static" --verbose

# With coverage report
npx jest tests/server.test.js --coverage
```

### Expected output
```
PASS tests/server.test.js
  detectSeparator()
    ✓ detects comma separator (2ms)
    ✓ detects semicolon separator (1ms)
    ✓ detects tab separator (1ms)
    ✓ detects pipe separator (1ms)
    ✓ defaults to comma for ambiguous input (1ms)
  looksLikeHeaderRow()
    ✓ detects source/target header (1ms)
    ✓ detects from/to header (1ms)
    ✓ returns false for data rows (1ms)
    ✓ is case-insensitive (1ms)
  POST /api/analyze
    ✓ returns 400 when no file uploaded (45ms)
    ✓ parses simple CSV and returns nodes + edges (180ms)
    ✓ nodes have required fields (120ms)
    ...
```

---

## Python Tests (pytest)

### What is tested

**Unit tests — model.py**
- `build_pyg_data()` — PyG data construction from DataFrames
- `GraphSAGE_NC` — forward pass shape, embedding shape, eval mode
- `GraphSAGE_LP` — encode/decode shapes, forward pass

**Integration tests — Flask API**
- `GET /health` — health check endpoint
- Demo endpoints — correct 400 before demo loaded
- `POST /gnn/train` — training on small graph
- User endpoints — correct 400 before training

**End-to-end tests**
- Full pipeline: build data → train NC → run inference → verify output

### Setup
```bash
cd graphml-studio/gnn_services
pip install pytest pytest-flask
```

### Run tests
```bash
# All tests
pytest tests/test_gnn_services.py -v

# Unit tests only (no Flask server needed)
pytest tests/test_gnn_services.py -v -k "TestBuildPygData or TestGraphSAGE"

# Integration tests only (Flask test client — no server needed)
pytest tests/test_gnn_services.py -v -k "TestHealth or TestDemo or TestUser"

# End-to-end test
pytest tests/test_gnn_services.py -v -k "TestEndToEnd"

# With coverage
pytest tests/test_gnn_services.py --cov=../gnn_services --cov-report=term-missing
```

### Expected output
```
tests/test_gnn_services.py::TestBuildPygData::test_basic_construction PASSED
tests/test_gnn_services.py::TestBuildPygData::test_feature_matrix_shape PASSED
tests/test_gnn_services.py::TestBuildPygData::test_edge_index_is_undirected PASSED
tests/test_gnn_services.py::TestBuildPygData::test_filters_unlabeled_nodes PASSED
tests/test_gnn_services.py::TestBuildPygData::test_handles_from_to_columns PASSED
tests/test_gnn_services.py::TestBuildPygData::test_label_encoder_is_contiguous PASSED
tests/test_gnn_services.py::TestGraphSAGENC::test_forward_pass_shape PASSED
...
```

---

## Test Coverage Summary

| Component | Tests | Type |
|---|---|---|
| CSV delimiter detection | 5 | Unit |
| Header row detection | 4 | Unit |
| Graph analysis endpoint | 11 | Integration |
| Edge prediction endpoint | 5 | Integration |
| Node prediction endpoint | 4 | Integration |
| PyG data builder | 6 | Unit |
| GraphSAGE NC model | 3 | Unit |
| GraphSAGE LP model | 3 | Unit |
| Flask health endpoint | 3 | Integration |
| Flask demo endpoints | 5 | Integration |
| Flask train endpoint | 5 | Integration |
| Training pipeline | 2 | Unit |
| Full NC pipeline | 1 | End-to-end |
| **Total** | **57** | |
