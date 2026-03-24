/**
 * Mock for networkanalysis-ts
 * Replaces the ESM package with a CommonJS stub for Jest testing.
 * The real package is used in production — this mock only runs during tests.
 */

class MockClustering {
  constructor({ nNodes }) {
    this.nNodes = nNodes;
    this._clusters = new Array(nNodes).fill(0);
  }
  getClusters() { return this._clusters; }
  orderClustersByNNodes() {}
}

class MockLeidenAlgorithm {
  setResolution() {}
  setNIterations() {}
  improveClustering(network, clustering) {
    // Assign alternating communities for mock
    for (let i = 0; i < clustering.nNodes; i++) {
      clustering._clusters[i] = i % 3;
    }
  }
  calcQuality() { return 0.5; }
}

class MockNetwork {
  constructor({ nNodes }) {
    this.nNodes = nNodes;
  }
  getNNodes() { return this.nNodes; }
  createNormalizedNetworkUsingAssociationStrength() {
    return new MockNetwork({ nNodes: this.nNodes });
  }
}

module.exports = {
  Network:        MockNetwork,
  Clustering:     MockClustering,
  LeidenAlgorithm: MockLeidenAlgorithm,
};
