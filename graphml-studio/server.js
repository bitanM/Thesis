const express = require('express');
const multer  = require('multer');
const csv     = require('csv-parser');
const http    = require('http');
const { Readable } = require('stream');
const Graph   = require('graphology');
const betweenness = require('graphology-metrics/centrality/betweenness');
const eigenvector = require('graphology-metrics/centrality/eigenvector');
const { degreeCentrality } = require('graphology-metrics/centrality/degree');
const louvain = require('graphology-communities-louvain');
const { Network, Clustering, LeidenAlgorithm } = require('networkanalysis-ts');

const app = express();
app.use(express.json({ limit: '50mb' }));
app.use(express.static('public'));

const upload = multer({ storage: multer.memoryStorage() });

const CENTRALITY_LIMITS = { maxNodes: 1500, maxEdges: 10000 };
const UI_LIMITS = { maxNodes: 5000, maxEdges: 15000 };
const AUTO_TRAIN_GNN = false;

function buildSampleGraph(fullGraph, edgesList) {
  const nodesWithDeg = fullGraph.mapNodes((node, attr) => ({
    id: node,
    degree: attr.degree || 0,
  }));
  nodesWithDeg.sort((a, b) => b.degree - a.degree);
  const sampleNodes = nodesWithDeg.slice(0, UI_LIMITS.maxNodes).map(n => n.id);
  const sampleSet = new Set(sampleNodes);

  let sampleEdges = edgesList.filter(e => sampleSet.has(e.from) && sampleSet.has(e.to));
  if (sampleEdges.length > UI_LIMITS.maxEdges) {
    const step = Math.ceil(sampleEdges.length / UI_LIMITS.maxEdges);
    sampleEdges = sampleEdges.filter((_, i) => i % step === 0);
  }

  const sampleGraph = new Graph({ type: 'undirected', multi: false, allowSelfLoops: false });
  sampleNodes.forEach(id => {
    const degree = fullGraph.getNodeAttribute(id, 'degree') || 0;
    sampleGraph.addNode(id, { degree });
  });
  sampleEdges.forEach(e => {
    if (!sampleGraph.hasEdge(e.from, e.to)) {
      sampleGraph.addEdge(e.from, e.to, { weight: e.weight });
    }
  });

  return { sampleGraph, sampleNodes, sampleEdges };
}

// ── Python GNN service config ──
const GNN_HOST = process.env.GNN_HOST || 'localhost';
const GNN_PORT = Number(process.env.GNN_PORT || 5001);
let   gnnAvailable = false;

// Check GNN service availability on startup and every 30s
function checkGNNService() {
  const req = http.request(
    { host: GNN_HOST, port: GNN_PORT, path: '/health', method: 'GET', timeout: 2000 },
    (res) => {
      gnnAvailable = res.statusCode === 200;
      if (gnnAvailable) console.log('[GNN] Python service is online ✓');
    }
  );
  req.on('error', () => { gnnAvailable = false; });
  req.on('timeout', () => { req.destroy(); gnnAvailable = false; });
  req.end();
}
function startGNNServiceMonitor() {
  checkGNNService();
  return setInterval(checkGNNService, 30000);
}

// ── GNN proxy helper ──
function proxyToGNN(path, method, body, res) {
  if (!gnnAvailable) {
    return res.status(503).json({
      error: 'GNN service unavailable. Start the Python Flask service on port 5001.',
      hint:  'cd gnn_services && python app.py'
    });
  }

  const bodyStr = body ? JSON.stringify(body) : '';
  const options = {
    host:    GNN_HOST,
    port:    GNN_PORT,
    path,
    method,
    headers: {
      'Content-Type':   'application/json',
      'Content-Length': Buffer.byteLength(bodyStr),
    },
  };

  const proxyReq = http.request(options, (proxyRes) => {
    let data = '';
    proxyRes.on('data', chunk => data += chunk);
    proxyRes.on('end', () => {
      try {
        res.status(proxyRes.statusCode).json(JSON.parse(data));
      } catch (e) {
        res.status(500).json({ error: 'Invalid response from GNN service.' });
      }
    });
  });

  proxyReq.on('error', (err) => {
    gnnAvailable = false;
    res.status(503).json({ error: 'GNN service connection failed: ' + err.message });
  });

  proxyReq.write(bodyStr);
  proxyReq.end();
}

// ── GNN service status ──
app.get('/api/gnn/status', (req, res) => {
  if (!gnnAvailable) {
    return res.json({ available: false, message: 'GNN service offline' });
  }
  const options = { host: GNN_HOST, port: GNN_PORT, path: '/health', method: 'GET' };
  const proxyReq = http.request(options, (proxyRes) => {
    let data = '';
    proxyRes.on('data', chunk => data += chunk);
    proxyRes.on('end', () => {
      try { res.json({ available: true, ...JSON.parse(data) }); }
      catch (e) { res.json({ available: true }); }
    });
  });
  proxyReq.on('error', () => res.json({ available: false }));
  proxyReq.end();
});

// ── Demo mode: load Farmer's Protest pre-trained model ──
app.post('/api/gnn/demo/load', (req, res) => {
  proxyToGNN('/gnn/demo/load', 'POST', {}, res);
});

// ── Demo mode: get t-SNE embeddings ──
app.get('/api/gnn/demo/embeddings', (req, res) => {
  if (!gnnAvailable) {
    return res.status(503).json({ error: 'GNN service unavailable.' });
  }
  const options = { host: GNN_HOST, port: GNN_PORT,
                    path: '/gnn/demo/embeddings', method: 'GET' };
  const proxyReq = http.request(options, (proxyRes) => {
    let data = '';
    proxyRes.on('data', chunk => data += chunk);
    proxyRes.on('end', () => {
      try { res.status(proxyRes.statusCode).json(JSON.parse(data)); }
      catch (e) { res.status(500).json({ error: 'Parse error' }); }
    });
  });
  proxyReq.on('error', (err) => res.status(503).json({ error: err.message }));
  proxyReq.end();
});

// ── Demo mode: get community details ──
app.get('/api/gnn/demo/communities', (req, res) => {
  if (!gnnAvailable) return res.status(503).json({ error: 'GNN service unavailable.' });
  const options = { host: GNN_HOST, port: GNN_PORT,
                    path: '/gnn/demo/communities', method: 'GET' };
  const proxyReq = http.request(options, (proxyRes) => {
    let data = '';
    proxyRes.on('data', chunk => data += chunk);
    proxyRes.on('end', () => {
      try { res.status(proxyRes.statusCode).json(JSON.parse(data)); }
      catch (e) { res.status(500).json({ error: 'Parse error' }); }
    });
  });
  proxyReq.on('error', (err) => res.status(503).json({ error: err.message }));
  proxyReq.end();
});

// ── Demo mode: node search ──
app.post('/api/gnn/demo/search', (req, res) => {
  proxyToGNN('/gnn/demo/search', 'POST', req.body, res);
});

// ── Demo mode: predict node community ──
app.post('/api/gnn/demo/predict-node', (req, res) => {
  proxyToGNN('/gnn/demo/predict-node', 'POST', req.body, res);
});

// ── User upload: train GNN on custom graph ──
app.post('/api/gnn/train', (req, res) => {
  proxyToGNN('/gnn/train', 'POST', req.body, res);
});

// ── User upload: get embeddings ──
app.get('/api/gnn/user/embeddings', (req, res) => {
  if (!gnnAvailable) return res.status(503).json({ error: 'GNN service unavailable.' });
  const options = { host: GNN_HOST, port: GNN_PORT,
                    path: '/gnn/user/embeddings', method: 'GET' };
  const proxyReq = http.request(options, (proxyRes) => {
    let data = '';
    proxyRes.on('data', chunk => data += chunk);
    proxyRes.on('end', () => {
      try { res.status(proxyRes.statusCode).json(JSON.parse(data)); }
      catch (e) { res.status(500).json({ error: 'Parse error' }); }
    });
  });
  proxyReq.on('error', (err) => res.status(503).json({ error: err.message }));
  proxyReq.end();
});

// ── User upload: predict node ──
app.post('/api/gnn/user/predict-node', (req, res) => {
  proxyToGNN('/gnn/user/predict-node', 'POST', req.body, res);
});


// ════════════════════════════════════════════════════
// EXISTING ENDPOINTS (unchanged from original)
// ════════════════════════════════════════════════════

function detectSeparator(firstLine) {
  const candidates = [',', ';', '\t', '|'];
  let best = { sep: ',', count: -1 };
  for (const sep of candidates) {
    const count = firstLine.split(sep).length - 1;
    if (count > best.count) best = { sep, count };
  }
  return best.sep;
}

function looksLikeHeaderRow(values) {
  const hints = new Set(['source','src','from','target','tgt','dst','to','weight','w']);
  return values.map(v => String(v||'').trim().toLowerCase()).some(v => hints.has(v));
}

function makeInputError(message) {
  const err = new Error(message);
  err.status = 400;
  return err;
}

function computeLeidenCommunities(nodeIds, edgesList) {
  const result = {};
  if (!nodeIds || nodeIds.length === 0) return result;
  if (!edgesList || edgesList.length === 0) {
    nodeIds.forEach((id, idx) => { result[id] = idx; });
    return result;
  }
  const idToIndex = new Map();
  nodeIds.forEach((id, idx) => idToIndex.set(id, idx));
  const sources = [], targets = [];
  edgesList.forEach(e => {
    const s = idToIndex.get(e.from), t = idToIndex.get(e.to);
    if (s !== undefined && t !== undefined) { sources.push(s); targets.push(t); }
  });
  if (sources.length === 0) {
    nodeIds.forEach((id, idx) => { result[id] = idx; }); return result;
  }
  const network = new Network({
    nNodes: nodeIds.length, setNodeWeightsToTotalEdgeWeights: true,
    edges: [sources, targets], sortedEdges: false, checkIntegrity: false
  });
  const normalizedNetwork = network.createNormalizedNetworkUsingAssociationStrength();
  const clusteringAlgorithm = new LeidenAlgorithm();
  clusteringAlgorithm.setResolution(0.2);
  clusteringAlgorithm.setNIterations(nodeIds.length > 2000 ? 10 : 50);
  let bestClustering = null, maxQuality = Number.NEGATIVE_INFINITY;
  const nStarts = nodeIds.length > 2000 ? 3 : 10;
  for (let i = 0; i < nStarts; i++) {
    const clustering = new Clustering({ nNodes: normalizedNetwork.getNNodes() });
    clusteringAlgorithm.improveClustering(normalizedNetwork, clustering);
    const quality = clusteringAlgorithm.calcQuality(normalizedNetwork, clustering);
    if (quality > maxQuality) { bestClustering = clustering; maxQuality = quality; }
  }
  if (!bestClustering) { nodeIds.forEach((id,idx) => { result[id]=idx; }); return result; }
  if (typeof bestClustering.orderClustersByNNodes === 'function')
    bestClustering.orderClustersByNNodes();
  let clustersArr = null;
  if (typeof bestClustering.getClusters === 'function') clustersArr = bestClustering.getClusters();
  else if (Array.isArray(bestClustering.clusters)) clustersArr = bestClustering.clusters;
  else if (Array.isArray(bestClustering._clusters)) clustersArr = bestClustering._clusters;
  if (!clustersArr || clustersArr.length < nodeIds.length) {
    nodeIds.forEach((id,idx) => { result[id]=idx; }); return result;
  }
  nodeIds.forEach((id, idx) => { result[id] = clustersArr[idx] ?? 0; });
  return result;
}

app.post('/api/analyze', upload.single('csv'), (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No file uploaded.' });
    console.log(`[+] Received: ${req.file.originalname} (${req.file.size} bytes)`);
    const graph = new Graph({ type: 'undirected', multi: false, allowSelfLoops: false });
    const edgesList = [];
    let rowsSeen = 0, firstRowChecked = false;
    const csvText  = req.file.buffer.toString('utf8');
    const firstLine = (csvText.split(/\r?\n/).find(l => l.trim().length > 0)) || '';
    const separator = detectSeparator(firstLine);
    const stream    = Readable.from([csvText]);
    stream.pipe(csv({ separator, headers: false }))
      .on('data', (row) => {
        rowsSeen++;
        const values = Object.values(row);
        if (!firstRowChecked) {
          firstRowChecked = true;
          if (looksLikeHeaderRow(values)) return;
        }
        if (values.length < 2) return;
        const src = String(values[0]).trim();
        const tgt = String(values[1]).trim();
        const wgt = parseFloat(values[2]) || 1;
        if (!src || !tgt || src === tgt) return;
        if (!graph.hasNode(src)) graph.addNode(src, { degree: 0 });
        if (!graph.hasNode(tgt)) graph.addNode(tgt, { degree: 0 });
        if (!graph.hasEdge(src, tgt)) {
          graph.addEdge(src, tgt, { weight: wgt });
          edgesList.push({ from: src, to: tgt, weight: wgt });
          graph.updateNodeAttribute(src, 'degree', d => (d||0)+1);
          graph.updateNodeAttribute(tgt, 'degree', d => (d||0)+1);
        }
      })
      .on('end', () => {
        try {
          if (rowsSeen === 0) throw makeInputError('No rows found in CSV.');
          if (graph.order === 0) throw makeInputError(
            `Graph is empty. Detected delimiter "${separator}".`
          );
          console.log(`[+] Graph: ${graph.order} nodes, ${graph.size} edges`);

          const useSample = graph.order > UI_LIMITS.maxNodes || graph.size > UI_LIMITS.maxEdges;
          let graphForMetrics = graph;
          let nodesForResponse = graph.nodes();
          let edgesForResponse = edgesList;
          if (useSample) {
            const sampled = buildSampleGraph(graph, edgesList);
            graphForMetrics = sampled.sampleGraph;
            nodesForResponse = sampled.sampleNodes;
            edgesForResponse = sampled.sampleEdges;
          }

          const communities = louvain(graphForMetrics);
          let leidenCommunities = {};
          try {
            leidenCommunities = computeLeidenCommunities(graphForMetrics.nodes(), edgesForResponse);
          } catch(e) {
            console.log('[!] Leiden failed, falling back to Louvain.');
            graphForMetrics.forEachNode(n => { leidenCommunities[n] = communities[n] ?? 0; });
          }
          const skipCentrality = graphForMetrics.order > CENTRALITY_LIMITS.maxNodes ||
                                 graphForMetrics.size > CENTRALITY_LIMITS.maxEdges;
          let betCent = {};
          const degCent = degreeCentrality(graphForMetrics);
          let eigCent = {};
          if (!skipCentrality) {
            betCent = betweenness(graphForMetrics);
            try { eigCent = eigenvector(graphForMetrics); }
            catch(e) { graphForMetrics.forEachNode(n => eigCent[n] = 0); }
          } else {
            console.log('[!] Graph too large for full centrality; skipping betweenness/eigenvector.');
            graphForMetrics.forEachNode(n => { betCent[n] = 0; eigCent[n] = 0; });
          }
          const nodesList = nodesForResponse.map(node => ({
            id:       node,
            degree:   graph.getNodeAttribute(node, 'degree') || 0,
            community: communities[node] ?? 0,
            communities: {
              louvain: communities[node] ?? 0,
              leiden:  leidenCommunities[node] ?? communities[node] ?? 0,
            },
            centrality: {
              betweenness: betCent[node] || 0,
              eigenvector: eigCent[node] || 0,
              degree:      degCent[node] || 0,
            },
          }));

          // ── Optional auto-train (disabled by default for performance) ──
          if (AUTO_TRAIN_GNN && gnnAvailable) {
            console.log('[GNN] Auto-training on uploaded graph...');
            const gnnPayload = JSON.stringify({ nodes: nodesList, edges: edgesForResponse, mode: 'fast' });
            const gnnReq = http.request({
              host: GNN_HOST, port: GNN_PORT, path: '/gnn/train', method: 'POST',
              headers: { 'Content-Type': 'application/json',
                         'Content-Length': Buffer.byteLength(gnnPayload) },
            }, (gnnRes) => {
              let d = '';
              gnnRes.on('data', c => d += c);
              gnnRes.on('end', () => console.log('[GNN] Training result:', d.slice(0, 200)));
            });
            gnnReq.on('error', err => console.log('[GNN] Training error:', err.message));
            gnnReq.write(gnnPayload);
            gnnReq.end();
          }

          res.json({
            nodes: nodesList,
            edges: edgesForResponse,
            meta: {
              sampled: useSample,
              total_nodes: graph.order,
              total_edges: graph.size,
              returned_nodes: nodesList.length,
              returned_edges: edgesForResponse.length,
              centralitySkipped: skipCentrality,
              centralityLimits: CENTRALITY_LIMITS,
              uiLimits: UI_LIMITS,
            }
          });
        } catch (algoError) {
          const status = algoError.status || 500;
          res.status(status).json({ error: algoError.message });
        }
      })
      .on('error', () => res.status(400).json({ error: 'Failed to parse CSV.' }));
  } catch (err) {
    res.status(500).json({ error: 'Internal Server Error.' });
  }
});

app.post('/api/predict-node', (req, res) => {
  // Route to real GNN if available, else toy fallback
  if (gnnAvailable) {
    return proxyToGNN('/gnn/user/predict-node', 'POST',
      { nodeId: req.body.newConnections?.[0] }, res);
  }
  // Toy fallback (original implementation)
  const { nodes, newConnections } = req.body || {};
  if (!Array.isArray(nodes) || !Array.isArray(newConnections))
    return res.status(400).json({ error: 'Invalid payload.' });
  const nodeById = new Map(nodes.map(n => [String(n.id), n]));
  const neighborCommunities = newConnections
    .map(c => nodeById.get(String(c).trim()))
    .map(n => Number(n ? n.community : NaN))
    .filter(n => Number.isFinite(n));
  if (neighborCommunities.length === 0)
    return res.json({ predictedCommunity: 0, confidence: 0, role: 'Isolated' });
  const counts = new Map();
  neighborCommunities.forEach(c => counts.set(c, (counts.get(c)||0)+1));
  let predComm = 0, maxVote = 0;
  counts.forEach((count, commId) => { if (count > maxVote) { maxVote=count; predComm=commId; } });
  res.json({
    predictedCommunity: predComm,
    confidence: ((maxVote/neighborCommunities.length)*100).toFixed(1),
    role: newConnections.length >= 3 ? 'Hub' : 'Member',
  });
});

app.post('/api/predict-edge', (req, res) => {
  // Route to GNN service if available (inductive link prediction)
  if (gnnAvailable && req.body && req.body.useGNN) {
    const { newNodeId, newConnections, topK } = req.body || {};
    return proxyToGNN('/gnn/user/predict-edge', 'POST',
      { nodeId: newNodeId, connections: newConnections, topK }, res);
  }
  const { nodes, edges, newNodeId, newConnections, topK } = req.body || {};
  if (!Array.isArray(nodes)||!Array.isArray(edges)||!Array.isArray(newConnections))
    return res.status(400).json({ error: 'Invalid payload.' });
  const validConns = new Set(newConnections.map(c => String(c).trim()).filter(Boolean));
  let predictions = [];
  nodes.forEach(node => {
    const nodeId = String(node.id);
    if (validConns.has(nodeId)) return;
    let shared = 0;
    edges.forEach(e => {
      if ((String(e.from)===nodeId && validConns.has(String(e.to))) ||
          (String(e.to)===nodeId && validConns.has(String(e.from)))) shared++;
    });
    const safeDeg = Math.max(Number(node.degree)||1, 1);
    const score = shared / Math.log(safeDeg + 2);
    if (Number.isFinite(score) && score > 0)
      predictions.push({ id: nodeId, score,
        prob: Math.min(95, Math.round(score*100+30)),
        estWeight: (1+score*0.5).toFixed(2) });
  });
  const k = Math.max(1, Math.min(20, Number(topK)||8));
  predictions.sort((a,b) => b.score - a.score);
  res.json({ predictions: predictions.slice(0, k) });
});

if (require.main === module) {
  startGNNServiceMonitor();
  const PORT = process.env.PORT || 3000;
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`GraphML Studio running on port ${PORT}`);
    console.log(`   GNN service expected at http://${GNN_HOST}:${GNN_PORT}`);
  });
}
module.exports = app;
