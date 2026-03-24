/**
 * GraphML Studio — Node.js Test Suite
 * Tests for server.js endpoints using Jest + Supertest
 *
 * Install dependencies:
 *   npm install --save-dev jest supertest
 *
 * Run:
 *   npx jest tests/server.test.js --verbose
 */

const request = require('supertest');

// ── We import the app without starting the listener ──
// Add this line to the bottom of server.js:
//   if (require.main === module) app.listen(3000, ...);
//   module.exports = app;
const app = require('../server');

// ════════════════════════════════════════════════
// UNIT TESTS — Helper functions
// ════════════════════════════════════════════════
describe('detectSeparator()', () => {
  // Re-export or inline the function for testing
  function detectSeparator(firstLine) {
    const candidates = [',', ';', '\t', '|'];
    let best = { sep: ',', count: -1 };
    for (const sep of candidates) {
      const count = firstLine.split(sep).length - 1;
      if (count > best.count) best = { sep, count };
    }
    return best.sep;
  }

  test('detects comma separator', () => {
    expect(detectSeparator('a,b,c')).toBe(',');
  });

  test('detects semicolon separator', () => {
    expect(detectSeparator('a;b;c')).toBe(';');
  });

  test('detects tab separator', () => {
    expect(detectSeparator('a\tb\tc')).toBe('\t');
  });

  test('detects pipe separator', () => {
    expect(detectSeparator('a|b|c')).toBe('|');
  });

  test('defaults to comma for ambiguous input', () => {
    expect(detectSeparator('abc')).toBe(',');
  });
});

describe('looksLikeHeaderRow()', () => {
  function looksLikeHeaderRow(values) {
    const hints = new Set(['source','src','from','target','tgt','dst','to','weight','w']);
    return values.map(v => String(v||'').trim().toLowerCase()).some(v => hints.has(v));
  }

  test('detects source/target header', () => {
    expect(looksLikeHeaderRow(['source', 'target', 'weight'])).toBe(true);
  });

  test('detects from/to header', () => {
    expect(looksLikeHeaderRow(['from', 'to'])).toBe(true);
  });

  test('returns false for data rows', () => {
    expect(looksLikeHeaderRow(['nodeA', 'nodeB', '1.5'])).toBe(false);
  });

  test('is case-insensitive', () => {
    expect(looksLikeHeaderRow(['SOURCE', 'TARGET'])).toBe(true);
  });
});

// ════════════════════════════════════════════════
// INTEGRATION TESTS — API Endpoints
// ════════════════════════════════════════════════

// ── /api/gnn/status ──
describe('GET /api/gnn/status', () => {
  test('returns JSON with available field', async () => {
    const res = await request(app).get('/api/gnn/status');
    expect(res.statusCode).toBe(200);
    expect(res.body).toHaveProperty('available');
    expect(typeof res.body.available).toBe('boolean');
  });
});

// ── /api/analyze ──
describe('POST /api/analyze', () => {
  test('returns 400 when no file uploaded', async () => {
    const res = await request(app).post('/api/analyze');
    expect(res.statusCode).toBe(400);
    expect(res.body).toHaveProperty('error');
  });

  test('parses simple CSV and returns nodes + edges', async () => {
    const csv = `source,target,weight\nA,B,1\nB,C,2\nA,C,3`;
    const res = await request(app)
      .post('/api/analyze')
      .attach('csv', Buffer.from(csv), 'test.csv');

    expect(res.statusCode).toBe(200);
    expect(res.body).toHaveProperty('nodes');
    expect(res.body).toHaveProperty('edges');
    expect(res.body.nodes.length).toBe(3);
    expect(res.body.edges.length).toBe(3);
  });

  test('nodes have required fields', async () => {
    const csv = `source,target\nX,Y\nY,Z`;
    const res = await request(app)
      .post('/api/analyze')
      .attach('csv', Buffer.from(csv), 'test.csv');

    expect(res.statusCode).toBe(200);
    const node = res.body.nodes[0];
    expect(node).toHaveProperty('id');
    expect(node).toHaveProperty('degree');
    expect(node).toHaveProperty('community');
    expect(node).toHaveProperty('communities');
    expect(node).toHaveProperty('centrality');
    expect(node.centrality).toHaveProperty('betweenness');
    expect(node.centrality).toHaveProperty('eigenvector');
    expect(node.centrality).toHaveProperty('degree');
  });

  test('edges have required fields', async () => {
    const csv = `source,target,weight\nA,B,1.5`;
    const res = await request(app)
      .post('/api/analyze')
      .attach('csv', Buffer.from(csv), 'test.csv');

    expect(res.statusCode).toBe(200);
    const edge = res.body.edges[0];
    expect(edge).toHaveProperty('from');
    expect(edge).toHaveProperty('to');
    expect(edge).toHaveProperty('weight');
  });

  test('handles CSV without weight column', async () => {
    const csv = `source,target\nA,B\nB,C`;
    const res = await request(app)
      .post('/api/analyze')
      .attach('csv', Buffer.from(csv), 'test.csv');

    expect(res.statusCode).toBe(200);
    expect(res.body.edges[0].weight).toBe(1);
  });

  test('handles headerless CSV', async () => {
    const csv = `A,B,1\nB,C,2\nC,D,3`;
    const res = await request(app)
      .post('/api/analyze')
      .attach('csv', Buffer.from(csv), 'test.csv');

    expect(res.statusCode).toBe(200);
    expect(res.body.nodes.length).toBe(4);
  });

  test('ignores self-loops', async () => {
    const csv = `source,target\nA,A\nA,B`;
    const res = await request(app)
      .post('/api/analyze')
      .attach('csv', Buffer.from(csv), 'test.csv');

    expect(res.statusCode).toBe(200);
    expect(res.body.edges.length).toBe(1);
  });

  test('ignores duplicate edges', async () => {
    const csv = `source,target\nA,B\nA,B\nA,B`;
    const res = await request(app)
      .post('/api/analyze')
      .attach('csv', Buffer.from(csv), 'test.csv');

    expect(res.statusCode).toBe(200);
    expect(res.body.edges.length).toBe(1);
  });

  test('returns 400 for empty CSV', async () => {
    const csv = ``;
    const res = await request(app)
      .post('/api/analyze')
      .attach('csv', Buffer.from(csv), 'empty.csv');

    expect(res.statusCode).toBe(400);
    expect(res.body).toHaveProperty('error');
  });

  test('handles semicolon-delimited CSV', async () => {
    const csv = `source;target;weight\nA;B;1\nB;C;2`;
    const res = await request(app)
      .post('/api/analyze')
      .attach('csv', Buffer.from(csv), 'semi.csv');

    expect(res.statusCode).toBe(200);
    expect(res.body.nodes.length).toBe(3);
  });

  test('communities object has louvain and leiden', async () => {
    const csv = `source,target\nA,B\nB,C\nC,D\nD,A`;
    const res = await request(app)
      .post('/api/analyze')
      .attach('csv', Buffer.from(csv), 'test.csv');

    expect(res.statusCode).toBe(200);
    const node = res.body.nodes[0];
    expect(node.communities).toHaveProperty('louvain');
    expect(node.communities).toHaveProperty('leiden');
  });

  test('handles large graph (100 nodes)', async () => {
    const rows = ['source,target,weight'];
    for (let i = 0; i < 100; i++) {
      rows.push(`node${i},node${(i+1)%100},${Math.random().toFixed(2)}`);
    }
    const res = await request(app)
      .post('/api/analyze')
      .attach('csv', Buffer.from(rows.join('\n')), 'large.csv');

    expect(res.statusCode).toBe(200);
    expect(res.body.nodes.length).toBe(100);
  }, 30000);
});

// ── /api/predict-edge ──
describe('POST /api/predict-edge', () => {
  const sampleNodes = [
    { id: 'A', degree: 3, community: 0 },
    { id: 'B', degree: 2, community: 0 },
    { id: 'C', degree: 4, community: 1 },
    { id: 'D', degree: 1, community: 1 },
  ];
  const sampleEdges = [
    { from: 'A', to: 'B', weight: 1 },
    { from: 'A', to: 'C', weight: 2 },
    { from: 'B', to: 'C', weight: 1.5 },
  ];

  test('returns predictions array', async () => {
    const res = await request(app)
      .post('/api/predict-edge')
      .send({ nodes: sampleNodes, edges: sampleEdges,
              newNodeId: 'E', newConnections: ['A', 'B'], topK: 3 });

    expect(res.statusCode).toBe(200);
    expect(res.body).toHaveProperty('predictions');
    expect(Array.isArray(res.body.predictions)).toBe(true);
  });

  test('predictions have required fields', async () => {
    const res = await request(app)
      .post('/api/predict-edge')
      .send({ nodes: sampleNodes, edges: sampleEdges,
              newNodeId: 'E', newConnections: ['A'], topK: 2 });

    expect(res.statusCode).toBe(200);
    if (res.body.predictions.length > 0) {
      const pred = res.body.predictions[0];
      expect(pred).toHaveProperty('id');
      expect(pred).toHaveProperty('prob');
      expect(pred).toHaveProperty('score');
    }
  });

  test('respects topK parameter', async () => {
    const res = await request(app)
      .post('/api/predict-edge')
      .send({ nodes: sampleNodes, edges: sampleEdges,
              newNodeId: 'E', newConnections: ['A'], topK: 2 });

    expect(res.statusCode).toBe(200);
    expect(res.body.predictions.length).toBeLessThanOrEqual(2);
  });

  test('returns 400 for invalid payload', async () => {
    const res = await request(app)
      .post('/api/predict-edge')
      .send({ invalid: true });

    expect(res.statusCode).toBe(400);
  });

  test('excludes existing connections from predictions', async () => {
    const res = await request(app)
      .post('/api/predict-edge')
      .send({ nodes: sampleNodes, edges: sampleEdges,
              newNodeId: 'E', newConnections: ['A', 'B'], topK: 10 });

    expect(res.statusCode).toBe(200);
    const predIds = res.body.predictions.map(p => p.id);
    expect(predIds).not.toContain('A');
    expect(predIds).not.toContain('B');
  });
});

// ── /api/predict-node ──
describe('POST /api/predict-node', () => {
  const sampleNodes = [
    { id: 'A', community: 0 },
    { id: 'B', community: 0 },
    { id: 'C', community: 1 },
  ];

  test('returns prediction with community and confidence', async () => {
    const res = await request(app)
      .post('/api/predict-node')
      .send({ nodes: sampleNodes, newConnections: ['A', 'B'] });

    expect(res.statusCode).toBe(200);
    expect(res.body).toHaveProperty('predictedCommunity');
    expect(res.body).toHaveProperty('confidence');
    expect(res.body).toHaveProperty('role');
  });

  test('returns isolated for empty connections', async () => {
    const res = await request(app)
      .post('/api/predict-node')
      .send({ nodes: sampleNodes, newConnections: [] });

    expect(res.statusCode).toBe(200);
    expect(res.body.role).toBe('Isolated');
    expect(res.body.confidence).toBe(0);
  });

  test('assigns Hub role for 3+ connections', async () => {
    const res = await request(app)
      .post('/api/predict-node')
      .send({ nodes: sampleNodes, newConnections: ['A', 'B', 'C'] });

    expect(res.statusCode).toBe(200);
    expect(res.body.role).toBe('Hub');
  });

  test('returns 400 for invalid payload', async () => {
    const res = await request(app)
      .post('/api/predict-node')
      .send({ bad: 'data' });

    expect(res.statusCode).toBe(400);
  });
});

// ── Static files ──
describe('Static file serving', () => {
  test('GET / serves index.html', async () => {
    const res = await request(app).get('/');
    expect(res.statusCode).toBe(200);
    expect(res.headers['content-type']).toMatch(/html/);
  });
});