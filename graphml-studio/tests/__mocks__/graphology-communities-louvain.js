/**
 * Mock for graphology-communities-louvain
 * Returns a simple community assignment for testing.
 */

function louvain(graph) {
  const communities = {};
  let i = 0;
  graph.forEachNode(node => {
    communities[node] = i % 3;
    i++;
  });
  return communities;
}

module.exports = louvain;
