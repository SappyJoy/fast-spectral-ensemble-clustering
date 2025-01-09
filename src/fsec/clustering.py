from sklearn.base import BaseEstimator, ClusterMixin
import dask.array as da

from fsec.qr_evd_mr import compute_evd_map_reduce

from .anchoring import BKHK_dask, compute_anchor_neighbors
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
        if not isinstance(X, da.Array):
            X = da.from_array(X, chunks=(1000, X.shape[1]))

        anchors, anchor_assignments = BKHK_dask(X, self.num_anchors)
        X = X.compute()

        # Step 2: Compute Anchor Neighbors
        K_prime = self.K_prime or 10 * self.K
        K_prime = min(K_prime, self.num_anchors - 1)
        anchor_neighbors = compute_anchor_neighbors(self.anchors, K_prime)

        # Step 3: Compute Sample-Anchor Similarities
        self.B = compute_sample_anchor_similarities(
            X, self.anchors, anchor_assignments, anchor_neighbors, self.K
        )

        # Step 4: Compute SVD
        self.U = compute_evd_map_reduce(self.B, self.n_components)

        # Step 5: Generate Base Clusterings
        self.base_clusterings = generate_base_clusterings(
            self.U, self.num_clusters_list, n_jobs=self.n_jobs
        )

        # Step 6: Build Bipartite Graph
        self.H = build_bipartite_graph(self.base_clusterings)

        # Step 7: Consensus Clustering
        self.labels_ = consensus_clustering(self.H, n_clusters=self.final_n_clusters)

        return self

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_

