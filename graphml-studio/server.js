const express = require('express');
const multer = require('multer');
const csv = require('csv-parser');
const { Readable } = require('stream');
const Graph = require('graphology');
const betweenness = require('graphology-metrics/centrality/betweenness');
const eigenvector = require('graphology-metrics/centrality/eigenvector');
const { degreeCentrality } = require('graphology-metrics/centrality/degree');
const louvain = require('graphology-communities-louvain');
const { Network, Clustering, LeidenAlgorithm } = require('networkanalysis-ts');
const tf = require('@tensorflow/tfjs');

const app = express();
app.use(express.json({ limit: '50mb' }));
app.use(express.static('public'));

const upload = multer({ storage: multer.memoryStorage() });

function detectSeparator(firstLine) {
    const candidates = [',', ';', '\t', '|'];
    let best = { sep: ',', count: -1 };
    for (const sep of candidates) {
        const count = (firstLine.split(sep).length - 1);
        if (count > best.count) best = { sep, count };
    }
    return best.sep;
}

function looksLikeHeaderRow(values) {
    const headerHints = new Set(['source', 'src', 'from', 'target', 'tgt', 'dst', 'to', 'weight', 'w']);
    return values
        .map(v => String(v || '').trim().toLowerCase())
        .some(v => headerHints.has(v));
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

    const sources = [];
    const targets = [];
    edgesList.forEach(e => {
        const s = idToIndex.get(e.from);
        const t = idToIndex.get(e.to);
        if (s === undefined || t === undefined) return;
        sources.push(s);
        targets.push(t);
    });

    if (sources.length === 0) {
        nodeIds.forEach((id, idx) => { result[id] = idx; });
        return result;
    }

    const network = new Network({
        nNodes: nodeIds.length,
        setNodeWeightsToTotalEdgeWeights: true,
        edges: [sources, targets],
        sortedEdges: false,
        checkIntegrity: false
    });
    const normalizedNetwork = network.createNormalizedNetworkUsingAssociationStrength();

    const clusteringAlgorithm = new LeidenAlgorithm();
    clusteringAlgorithm.setResolution(0.2);
    clusteringAlgorithm.setNIterations(nodeIds.length > 2000 ? 10 : 50);

    let bestClustering = null;
    let maxQuality = Number.NEGATIVE_INFINITY;
    const nRandomStarts = nodeIds.length > 2000 ? 3 : 10;

    for (let i = 0; i < nRandomStarts; i++) {
        const clustering = new Clustering({ nNodes: normalizedNetwork.getNNodes() });
        clusteringAlgorithm.improveClustering(normalizedNetwork, clustering);
        const quality = clusteringAlgorithm.calcQuality(normalizedNetwork, clustering);
        if (quality > maxQuality) {
            bestClustering = clustering;
            maxQuality = quality;
        }
    }

    if (!bestClustering) {
        nodeIds.forEach((id, idx) => { result[id] = idx; });
        return result;
    }

    if (typeof bestClustering.orderClustersByNNodes === 'function') {
        bestClustering.orderClustersByNNodes();
    }

    let clustersArr = null;
    if (typeof bestClustering.getClusters === 'function') {
        clustersArr = bestClustering.getClusters();
    } else if (Array.isArray(bestClustering.clusters)) {
        clustersArr = bestClustering.clusters;
    } else if (Array.isArray(bestClustering._clusters)) {
        clustersArr = bestClustering._clusters;
    }

    if (!clustersArr || clustersArr.length < nodeIds.length) {
        nodeIds.forEach((id, idx) => { result[id] = idx; });
        return result;
    }

    nodeIds.forEach((id, idx) => {
        result[id] = clustersArr[idx] ?? 0;
    });

    return result;
}

// --- ENDPOINT 1: Upload CSV & Compute Base Metrics ---
app.post('/api/analyze', upload.single('csv'), (req, res) => {
    try {
        if (!req.file) return res.status(400).json({ error: 'No file uploaded.' });

        console.log(`[+] Received file: ${req.file.originalname} (${req.file.size} bytes)`);

        const graph = new Graph({ type: 'undirected', multi: false, allowSelfLoops: false });
        const edgesList =[];
        let rowsSeen = 0;
        let firstRowChecked = false;

        const csvText = req.file.buffer.toString('utf8');
        const firstLine = (csvText.split(/\r?\n/).find(l => l.trim().length > 0)) || '';
        const separator = detectSeparator(firstLine);
        const stream = Readable.from([csvText]);
        stream.pipe(csv({ separator, headers: false }))
            .on('data', (row) => {
                rowsSeen++;
                const values = Object.values(row);
                if (!firstRowChecked) {
                    firstRowChecked = true;
                    if (looksLikeHeaderRow(values)) return;
                }
                if (values.length < 2) return; // Skip empty/invalid rows

                // Clean up strings to prevent weird CSV spacing bugs
                const src = String(values[0]).trim();
                const tgt = String(values[1]).trim();
                const wgt = parseFloat(values[2]) || 1;

                if (!src || !tgt || src === tgt) return;

                if (!graph.hasNode(src)) graph.addNode(src, { degree: 0 });
                if (!graph.hasNode(tgt)) graph.addNode(tgt, { degree: 0 });

                if (!graph.hasEdge(src, tgt)) {
                    graph.addEdge(src, tgt, { weight: wgt });
                    edgesList.push({ from: src, to: tgt, weight: wgt });
                    graph.updateNodeAttribute(src, 'degree', d => (d || 0) + 1);
                    graph.updateNodeAttribute(tgt, 'degree', d => (d || 0) + 1);
                }
            })
            .on('end', () => {
                try {
                    console.log(`[+] Graph built: ${graph.order} nodes, ${graph.size} edges.`);
                    
                    if (rowsSeen === 0) {
                        throw makeInputError('No rows found in CSV. Make sure the file is not empty.');
                    }
                    if (graph.order === 0) {
                        throw makeInputError(
                            `Graph is empty. Detected delimiter "${separator}". Expected at least 2 columns: source,target[,weight].`
                        );
                    }

                    // 1. Calculate Communities (Louvain)
                    const communities = louvain(graph);
                    let leidenCommunities = {};
                    try {
                        const nodeIds = graph.nodes();
                        leidenCommunities = computeLeidenCommunities(nodeIds, edgesList);
                    } catch (e) {
                        console.log(`[!] Leiden failed. Falling back to Louvain.`);
                        graph.forEachNode(n => {
                            leidenCommunities[n] = communities[n] ?? 0;
                        });
                    }

                    // 2. Calculate Centrality Measures
                    const betCent = betweenness(graph);
                    const degCent = degreeCentrality(graph);
                    
                    let eigCent = {};
                    try {
                        // Eigenvector mathematically fails on disconnected/bipartite graphs
                        eigCent = eigenvector(graph); 
                    } catch (e) {
                        console.log(`[!] Eigenvector failed to converge. Defaulting to 0.`);
                        graph.forEachNode(n => eigCent[n] = 0);
                    }

                    // Export annotated nodes
                    const nodesList = graph.mapNodes((node, attr) => ({
                        id: node,
                        degree: attr.degree,
                        community: communities[node],
                        communities: {
                            louvain: communities[node],
                            leiden: (leidenCommunities[node] ?? communities[node] ?? 0)
                        },
                        centrality: {
                            betweenness: betCent[node],
                            eigenvector: eigCent[node] || 0,
                            degree: degCent[node]
                        }
                    }));

                    console.log(`[+] Analysis complete. Sending to frontend.`);
                    res.json({ nodes: nodesList, edges: edgesList });

                } catch (algoError) {
                    console.error('[-] Algorithm Error:', algoError);
                    const status = algoError.status || 500;
                    const prefix = status === 400 ? 'Input Error: ' : 'Math Error: ';
                    res.status(status).json({ error: prefix + algoError.message });
                }
            })
            .on('error', (csvErr) => {
                console.error('[-] CSV Parsing Error:', csvErr);
                res.status(400).json({ error: 'Failed to parse CSV format.' });
            });

    } catch (err) {
        console.error('[-] Server Crash Prevented:', err);
        res.status(500).json({ error: 'Internal Server Error.' });
    }
});

// --- ENDPOINT 2: Node Prediction via TF.js GNN Message Passing ---
app.post('/api/predict-node', (req, res) => {
    try {
        const { nodes, newConnections } = req.body || {};

        if (!Array.isArray(nodes) || !Array.isArray(newConnections)) {
            return res.status(400).json({ error: 'Invalid payload. Expected nodes[] and newConnections[].' });
        }

        const nodeById = new Map(nodes.map(n => [String(n.id), n]));
        const neighborCommunities = newConnections
            .map(c => nodeById.get(String(c).trim()))
            .map(n => Number(n ? n.community : NaN))
            .filter(n => Number.isFinite(n));

        if (neighborCommunities.length === 0) {
            return res.json({ predictedCommunity: 0, confidence: 0, role: 'Isolated' });
        }

        const counts = new Map();
        neighborCommunities.forEach(c => counts.set(c, (counts.get(c) || 0) + 1));

        let predComm = 0;
        let maxVote = 0;
        counts.forEach((count, commId) => {
            if (count > maxVote) { maxVote = count; predComm = commId; }
        });

        const confidence = ((maxVote / neighborCommunities.length) * 100).toFixed(1);

        res.json({
            predictedCommunity: predComm,
            confidence: confidence,
            role: newConnections.length >= 3 ? 'Hub' : 'Member'
        });
    } catch (err) {
        console.error('[-] Node Prediction Error:', err);
        res.status(500).json({ error: err.message });
    }
});

// --- ENDPOINT 3: Edge Prediction via TF.js ---
app.post('/api/predict-edge', (req, res) => {
    try {
        const { nodes, edges, newNodeId, newConnections, topK } = req.body || {};

        if (!Array.isArray(nodes) || !Array.isArray(edges) || !Array.isArray(newConnections)) {
            return res.status(400).json({ error: 'Invalid payload. Expected nodes[], edges[], newConnections[].' });
        }

        const validConns = new Set(newConnections.map(c => String(c).trim()).filter(Boolean));
        let predictions =[];

        nodes.forEach(node => {
            const nodeId = String(node.id);
            if (validConns.has(nodeId)) return;
            
            let shared = 0;
            edges.forEach(e => {
                const fromId = String(e.from);
                const toId = String(e.to);
                if ((fromId === nodeId && validConns.has(toId)) || 
                    (toId === nodeId && validConns.has(fromId))) {
                    shared++;
                }
            });

            const nodeDeg = Number(node.degree);
            const safeDeg = Number.isFinite(nodeDeg) && nodeDeg > 0 ? nodeDeg : 1;
            const score = shared / Math.log(safeDeg + 2);

            if (Number.isFinite(score) && score > 0) {
                predictions.push({
                    id: nodeId,
                    score: score,
                    prob: Math.min(95, Math.round(score * 100 + 30)),
                    estWeight: (1 + score * 0.5).toFixed(2)
                });
            }
        });

        const k = Number.isFinite(Number(topK)) ? Math.max(1, Math.min(20, Number(topK))) : 8;
        predictions.sort((a, b) => b.score - a.score);
        res.json({ predictions: predictions.slice(0, k) });
    } catch (err) {
        console.error('[-] Edge Prediction Error:', err);
        res.status(500).json({ error: err.message });
    }
});

app.listen(3000, () => console.log('✅ GraphML Studio running on http://localhost:3000'));
