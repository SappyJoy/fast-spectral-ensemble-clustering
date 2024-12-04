from sklearn.base import BaseEstimator, ClusterMixin

from .anchoring import BKHK, DBSCAN_anchor_selection, compute_anchor_neighbors
from .ensemble import (build_bipartite_graph, consensus_clustering,
                       generate_base_clusterings)
from .similarity import compute_sample_anchor_similarities
from .spectral import compute_svd


class FSEC(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        num_anchors=50,
        K_prime=None,
        K=5,
        n_components=2,
        num_clusters_list=None,
        final_n_clusters=2,
        n_jobs=-1,
        anchor_method='BKHK',
        use_mini_batch=False,
        dbscan_eps=0.5,         # DBSCAN-specific parameters
        dbscan_min_samples=5
    ):
        self.num_anchors = num_anchors
        self.K_prime = K_prime
        self.K = K
        self.n_components = n_components
        self.num_clusters_list = num_clusters_list or [final_n_clusters]
        self.final_n_clusters = final_n_clusters
        self.n_jobs = n_jobs
        self.anchor_method = anchor_method
        self.use_mini_batch = use_mini_batch
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

    def fit(self, X, y=None):
        # Step 1: Anchor Selection
        if self.anchor_method == 'BKHK':
            anchors, anchor_assignments = BKHK(
                X, self.num_anchors, use_mini_batch=self.use_mini_batch
            )
        elif self.anchor_method == 'DBSCAN':
            anchors, anchor_assignments = DBSCAN_anchor_selection(
                X,
                self.num_anchors,
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples
            )
        else:
            raise ValueError(f"Unknown anchor_method: {self.anchor_method}")

        # Step 2: Compute Anchor Neighbors
        K_prime = self.K_prime or 10 * self.K
        K_prime = min(K_prime, self.num_anchors - 1)
        anchor_neighbors = compute_anchor_neighbors(anchors, K_prime)

        # Step 3: Compute Sample-Anchor Similarities
        W = compute_sample_anchor_similarities(
            X, anchors, anchor_assignments, anchor_neighbors, self.K
        )

        # Step 4: Compute SVD
        U = compute_svd(W, self.n_components)

        # Step 5: Generate Base Clusterings
        base_clusterings = generate_base_clusterings(
            U, self.num_clusters_list, n_jobs=self.n_jobs
        )

        # Step 6: Build Bipartite Graph
        H = build_bipartite_graph(base_clusterings)

        # Step 7: Consensus Clustering
        self.labels_ = consensus_clustering(H, n_clusters=self.final_n_clusters)

        return self

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_

