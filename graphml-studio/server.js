/*
Copyright (C) 2026 Bitan Majumder

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
const express = require('express');
const multer  = require('multer');
const csv     = require('csv-parser');
const http    = require('http');
const https   = require('https');
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
const RAW_GNN_URL = process.env.GNN_URL || '';
let GNN_URL = RAW_GNN_URL;
if (RAW_GNN_URL && !/^[a-zA-Z]+:\/\//.test(RAW_GNN_URL)) {
  const isLocal = /^localhost(:|$)|^127\.0\.0\.1(:|$)/.test(RAW_GNN_URL);
  GNN_URL = `${isLocal ? 'http' : 'https'}://${RAW_GNN_URL}`;
}
const GNN_HOST = process.env.GNN_HOST || 'localhost';
const GNN_PORT = Number(process.env.GNN_PORT || 5001);
const GNN_TIMEOUT_MS = Number(process.env.GNN_TIMEOUT_MS || (GNN_URL ? 15000 : 2000));
const GNN_WAKE_TIMEOUT_MS = Number(process.env.GNN_WAKE_TIMEOUT_MS || 10000);
const GNN_PUBLIC_URL = process.env.GNN_PUBLIC_URL || GNN_URL;
let   gnnAvailable = false;

// Check GNN service availability on startup and every 30s
function buildGNNOptions(path, method, bodyStr = '') {
  if (GNN_URL) {
    const u = new URL(GNN_URL);
    const basePath = u.pathname === '/' ? '' : u.pathname.replace(/\/$/, '');
    return {
      protocol: u.protocol,
      host: u.hostname,
      port: u.port || (u.protocol === 'https:' ? 443 : 80),
      path: basePath + path,
      method,
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(bodyStr),
      },
      timeout: GNN_TIMEOUT_MS,
    };
  }
  return {
    protocol: 'http:',
    host: GNN_HOST,
    port: GNN_PORT,
    path,
    method,
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(bodyStr),
    },
    timeout: GNN_TIMEOUT_MS,
  };
}

function gnnRequest(options, onResponse) {
  const mod = options.protocol === 'https:' ? https : http;
  return mod.request(options, onResponse);
}

function probeGNN(timeoutMs) {
  return new Promise((resolve) => {
    const options = buildGNNOptions('/health', 'GET', '');
    if (timeoutMs) options.timeout = timeoutMs;
    const req = gnnRequest(options, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        resolve({ ok: res.statusCode === 200, statusCode: res.statusCode, data });
      });
    });
    req.on('error', (err) => resolve({ ok: false, error: err }));
    req.on('timeout', () => {
      req.destroy();
      resolve({ ok: false, error: new Error('timeout') });
    });
    req.end();
  });
}

// ── GNN proxy helper ──
function proxyToGNN(path, method, body, res, fallbackHandler = null) {
  const bodyStr = body ? JSON.stringify(body) : '';
  const options = buildGNNOptions(path, method, bodyStr);

  const proxyReq = gnnRequest(options, (proxyRes) => {
    let data = '';
    proxyRes.on('data', chunk => data += chunk);
    proxyRes.on('end', () => {
      if ((proxyRes.statusCode || 0) >= 400) {
        if (typeof fallbackHandler === 'function') {
          gnnAvailable = false;
          return fallbackHandler();
        }
        res.status(proxyRes.statusCode).json({
          error: 'GNN service request failed.',
          status: proxyRes.statusCode,
        });
        return;
      }
      gnnAvailable = true;
      try {
        res.status(proxyRes.statusCode).json(JSON.parse(data));
      } catch (e) {
        if (typeof fallbackHandler === 'function') {
          gnnAvailable = false;
          return fallbackHandler();
        }
        res.status(500).json({ error: 'Invalid response from GNN service.' });
      }
    });
  });

  proxyReq.on('error', (err) => {
    gnnAvailable = false;
    if (typeof fallbackHandler === 'function') {
      return fallbackHandler();
    }
    res.status(503).json({ error: 'GNN service connection failed: ' + err.message });
  });

  proxyReq.write(bodyStr);
  proxyReq.end();
}

function predictNodeMajorityFallback(payload, res) {
  const { nodes, newConnections, connections } = payload || {};
  const rawConnections = Array.isArray(newConnections)
    ? newConnections
    : (Array.isArray(connections) ? connections : []);
  if (!Array.isArray(nodes) || (!Array.isArray(newConnections) && !Array.isArray(connections))) {
    return res.status(400).json({ error: 'Invalid payload.' });
  }

  const nodeById = new Map(nodes.map(n => [String(n.id), n]));
  const neighborCommunities = rawConnections
    .map(c => nodeById.get(String(c).trim()))
    .map(n => Number(n ? n.community : NaN))
    .filter(n => Number.isFinite(n));

  if (neighborCommunities.length === 0) {
    return res.json({
      predictedCommunity: 0,
      confidence: 0,
      role: 'Isolated',
      heuristic: 'majority_vote',
    });
  }

  const counts = new Map();
  neighborCommunities.forEach(c => counts.set(c, (counts.get(c) || 0) + 1));
  let predComm = 0;
  let maxVote = 0;
  counts.forEach((count, commId) => {
    if (count > maxVote) {
      maxVote = count;
      predComm = commId;
    }
  });

  return res.json({
    predictedCommunity: predComm,
    confidence: ((maxVote / neighborCommunities.length) * 100).toFixed(1),
    role: rawConnections.length >= 3 ? 'Hub' : 'Member',
    heuristic: 'majority_vote',
  });
}

function predictEdgeAdamicAdarFallback(payload, res) {
  const { nodes, edges, newNodeId, newConnections, connections, topK } = payload || {};
  const rawConnections = Array.isArray(newConnections)
    ? newConnections
    : (Array.isArray(connections) ? connections : []);
  if (!Array.isArray(nodes) || !Array.isArray(edges) || !Array.isArray(rawConnections)) {
    return res.status(400).json({ error: 'Invalid payload.' });
  }

  const seedSet = new Set(rawConnections.map(c => String(c).trim()).filter(Boolean));
  const candidateId = String(newNodeId || '').trim();
  const neighborMap = new Map();
  const degreeMap = new Map();

  nodes.forEach(node => {
    const id = String(node.id);
    neighborMap.set(id, new Set());
    degreeMap.set(id, Math.max(Number(node.degree) || 0, 0));
  });

  edges.forEach(e => {
    const from = String(e.from).trim();
    const to = String(e.to).trim();
    if (!from || !to || from === to) return;
    if (!neighborMap.has(from)) neighborMap.set(from, new Set());
    if (!neighborMap.has(to)) neighborMap.set(to, new Set());
    neighborMap.get(from).add(to);
    neighborMap.get(to).add(from);
    degreeMap.set(from, (degreeMap.get(from) || 0) + 1);
    degreeMap.set(to, (degreeMap.get(to) || 0) + 1);
  });

  const predictions = [];
  nodes.forEach(node => {
    const nodeId = String(node.id);
    if (!nodeId || seedSet.has(nodeId) || nodeId === candidateId) return;

    const neighbors = neighborMap.get(nodeId) || new Set();
    let score = 0;
    seedSet.forEach(seedId => {
      if (!neighbors.has(seedId)) return;
      const deg = Math.max(Number(degreeMap.get(seedId)) || 0, 1);
      score += 1 / Math.log(deg + 1);
    });

    if (Number.isFinite(score) && score > 0) {
      predictions.push({
        id: nodeId,
        score,
        prob: Math.min(99, Math.round(score * 100)),
        estWeight: (1 + score * 0.5).toFixed(2),
      });
    }
  });

  const k = Math.max(1, Math.min(20, Number(topK) || 8));
  predictions.sort((a, b) => b.score - a.score);
  return res.json({
    predictions: predictions.slice(0, k),
    heuristic: 'adamic_adar',
  });
}

// ── GNN service status ──
app.get('/api/gnn/status', async (req, res) => {
  try {
    const result = await probeGNN();
    if (!result.ok) {
      gnnAvailable = false;
      return res.json({
        available: false,
        message: 'GNN service offline',
        gnn_url: GNN_PUBLIC_URL || null,
        status: result.statusCode || null,
        error: result.error ? String(result.error.message || result.error) : null,
      });
    }
    gnnAvailable = true;
    try {
      const payload = JSON.parse(result.data || '{}');
      return res.json({ available: true, gnn_url: GNN_PUBLIC_URL || null, ...payload });
    } catch (e) {
      return res.json({ available: true, gnn_url: GNN_PUBLIC_URL || null });
    }
  } catch (e) {
    gnnAvailable = false;
    return res.json({ available: false, message: 'GNN service offline', gnn_url: GNN_PUBLIC_URL || null });
  }
});

// Wake GNN (forces a live health probe to warm the service)
async function handleGNNWake(req, res) {
  try {
    const result = await probeGNN(GNN_WAKE_TIMEOUT_MS);
    if (!result.ok) {
      gnnAvailable = false;
      return res.json({
        available: false,
        warming: true,
        message: 'Warming GNN service',
        gnn_url: GNN_PUBLIC_URL || null,
        status: result.statusCode || null,
        error: result.error ? String(result.error.message || result.error) : null,
      });
    }
    gnnAvailable = true;
    try {
      const payload = JSON.parse(result.data || '{}');
      return res.json({ available: true, gnn_url: GNN_PUBLIC_URL || null, ...payload });
    } catch (e) {
      return res.json({ available: true, gnn_url: GNN_PUBLIC_URL || null });
    }
  } catch (e) {
    gnnAvailable = false;
    return res.json({ available: false, message: 'GNN service offline', gnn_url: GNN_PUBLIC_URL || null });
  }
}

app.post('/api/gnn/wake', handleGNNWake);
app.get('/api/gnn/wake', handleGNNWake);

// ── User upload: train GNN on custom graph ──
app.post('/api/gnn/train', (req, res) => {
  proxyToGNN('/gnn/train', 'POST', req.body, res);
});

// ── User upload: get embeddings ──
app.get('/api/gnn/user/embeddings', (req, res) => {
  proxyToGNN('/gnn/user/embeddings', 'GET', null, res);
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
  const fallback = () => predictNodeMajorityFallback(req.body, res);
  if (req.body && req.body.useGNN === true && gnnAvailable) {
    return proxyToGNN('/gnn/user/predict-node', 'POST', req.body, res, fallback);
  }
  return fallback();
});

app.post('/api/predict-edge', (req, res) => {
  const fallback = () => predictEdgeAdamicAdarFallback(req.body, res);
  const shouldUseGNN = req.body && req.body.useGNN === true;
  if (gnnAvailable && shouldUseGNN) {
    return proxyToGNN('/gnn/user/predict-edge', 'POST', req.body, res, fallback);
  }
  return fallback();
});

if (require.main === module) {
  const PORT = process.env.PORT || 3000;
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`GraphML Studio running on port ${PORT}`);
    console.log(`   GNN service expected at http://${GNN_HOST}:${GNN_PORT}`);
  });
}
module.exports = app;
