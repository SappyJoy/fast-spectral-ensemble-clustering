import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_data(X, y, title="Data Distribution", xlabel="Component 1", ylabel="Component 2"):
    plt.figure(figsize=(8, 6))
    if y is not None:
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', s=10)
        plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(X[:, 0], X[:, 1], s=10, color='grey')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_anchors(X_embedded, anchors_embedded, title="Data with Anchors"):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], s=10, color='blue', alpha=0.5, label='Data Points')
    plt.scatter(anchors_embedded[:, 0], anchors_embedded[:, 1], c='red', marker='X', s=100, label='Anchors')
    plt.legend()
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

def plot_anchor_neighbors(anchors_pca, anchor_neighbors, title="Anchor Neighbors"):
    """
    Plot anchors and their neighbor connections using PCA-reduced data.
    
    Parameters:
    - anchors_pca: PCA-reduced coordinates of anchors (numpy array of shape (n_anchors, 2))
    - anchor_neighbors: Array-like of shape (n_anchors, K_prime) containing neighbor indices for each anchor
    - title: Title of the plot (string)
    """
    plt.figure(figsize=(8, 6))
    
    # Plot all anchors
    plt.scatter(anchors_pca[:, 0], anchors_pca[:, 1], 
                c='red', marker='X', s=100, label='Anchors')
    
    # Draw lines between anchors and their neighbors
    for i, neighbors in enumerate(anchor_neighbors):
        for neighbor in neighbors:
            plt.plot([anchors_pca[i, 0], anchors_pca[neighbor, 0]],
                     [anchors_pca[i, 1], anchors_pca[neighbor, 1]],
                     'k-', lw=0.5, alpha=0.5)
    
    plt.legend()
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

def plot_similarity_matrix(W, subset=100, title="Similarity Matrix Subset"):
    W_dense = W[:subset, :subset].toarray()
    plt.figure(figsize=(8, 6))
    sns.heatmap(W_dense, cmap='viridis')
    plt.title(title)
    plt.xlabel("Anchors")
    plt.ylabel("Samples")
    plt.show()

def plot_spectral_embedding(U_tsne, y=None, title="Spectral Embedding"):
    plt.figure(figsize=(8, 6))
    if y is not None:
        scatter = plt.scatter(U_tsne[:, 0], U_tsne[:, 1], c=y, cmap='tab10', s=10)
        plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(U_tsne[:, 0], U_tsne[:, 1], s=10, color='grey')
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()

def plot_singular_values(Sigma, title="Singular Values (Scree Plot)"):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(Sigma)+1), Sigma, 'o-', color='blue')
    plt.title(title)
    plt.xlabel("Component Number")
    plt.ylabel("Singular Value")
    plt.grid(True)
    plt.show()


def plot_spectral_embedding_direct(U, y=None, title="Spectral Embedding (Direct Plot)"):
    plt.figure(figsize=(8, 6))
    if y is not None:
        scatter = plt.scatter(U[:, 0], U[:, 1], c=y, cmap='tab10', s=10)
        plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(U[:, 0], U[:, 1], s=10, color='grey')
    plt.title(title)
    plt.xlabel("Spectral Component 1")
    plt.ylabel("Spectral Component 2")
    plt.show()

def plot_final_clustering_direct(U, y, final_labels, title_true="True Labels", title_final="Final Clustering Results"):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # True Labels
    if y is not None:
        scatter = axes[0].scatter(U[:, 0], U[:, 1], c=y, cmap='tab10', s=10)
        axes[0].legend(*scatter.legend_elements(), title="True Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].set_title(title_true + " (Spectral Embedding)")
        axes[0].set_xlabel("Spectral Component 1")
        axes[0].set_ylabel("Spectral Component 2")
    else:
        axes[0].scatter(U[:, 0], U[:, 1], s=10, color='grey')
        axes[0].set_title("No True Labels Available")
        axes[0].set_xlabel("Spectral Component 1")
        axes[0].set_ylabel("Spectral Component 2")
    
    # Final Clusters
    scatter = axes[1].scatter(U[:, 0], U[:, 1], c=final_labels, cmap='tab10', s=10)
    axes[1].legend(*scatter.legend_elements(), title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].set_title(title_final + " (Spectral Embedding)")
    axes[1].set_xlabel("Spectral Component 1")
    axes[1].set_ylabel("Spectral Component 2")
    
    plt.tight_layout()
    plt.show()


def plot_spectral_embedding_pca(U, y=None, title="Spectral Embedding (PCA Reduced)"):
    pca = PCA(n_components=2, random_state=42)
    U_pca = pca.fit_transform(U)
    
    plt.figure(figsize=(8, 6))
    if y is not None:
        scatter = plt.scatter(U_pca[:, 0], U_pca[:, 1], c=y, cmap='tab10', s=10)
        plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(U_pca[:, 0], U_pca[:, 1], s=10, color='grey')
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

def plot_base_clusterings(U_tsne, base_clusterings, num_visualize=3, num_clusters_list=[8,9,10], title_prefix="Base Clustering"):
    plt.figure(figsize=(15, 4))
    for i in range(num_visualize):
        plt.subplot(1, num_visualize, i+1)
        labels = base_clusterings[i]
        plt.scatter(U_tsne[:, 0], U_tsne[:, 1], c=labels, cmap='tab10', s=10)
        plt.title(f"{title_prefix} {i+1} (k={num_clusters_list[i]})")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
    plt.tight_layout()
    plt.show()

def plot_base_clusterings_direct(U, base_clusterings, num_visualize=3, num_clusters_list=[2,3,4], title_prefix="Base Clustering"):
    plt.figure(figsize=(15, 4))
    for i in range(num_visualize):
        plt.subplot(1, num_visualize, i+1)
        labels = base_clusterings[i]
        plt.scatter(U[:, 0], U[:, 1], c=labels, cmap='tab10', s=10)
        plt.title(f"{title_prefix} {i+1} (k={num_clusters_list[i]})")
        plt.xlabel("Spectral Component 1")
        plt.ylabel("Spectral Component 2")
    plt.tight_layout()
    plt.show()

def plot_cluster_membership_heatmap(base_clusterings, title="Cluster Membership Heatmap"):
    from sklearn.metrics import pairwise_distances
    from scipy.cluster.hierarchy import linkage, dendrogram
    
    # Create a binary matrix indicating cluster memberships
    n_samples = base_clusterings[0].shape[0]
    n_clusterings = len(base_clusterings)
    cluster_membership = np.zeros((n_samples, n_clusterings), dtype=int)
    
    for i, labels in enumerate(base_clusterings):
        cluster_membership[:, i] = labels
    
    # Compute pairwise similarity
    similarity = pairwise_distances(cluster_membership, metric='hamming')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity, cmap='viridis')
    plt.title(title)
    plt.xlabel("Clusterings")
    plt.ylabel("Clusterings")
    plt.show()

def plot_degree_distribution(H, title="Degree Distribution of Clusters in Bipartite Graph"):
    cluster_degrees = np.array(H.sum(axis=0)).flatten()
    plt.figure(figsize=(10, 6))
    sns.histplot(cluster_degrees, bins=50, kde=True)
    plt.title(title)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()

def plot_final_clustering_direct(U, y, final_labels, title_true="True Labels", title_final="Final Clustering Results"):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # True Labels
    if y is not None:
        scatter = axes[0].scatter(U[:, 0], U[:, 1], c=y, cmap='tab10', s=10)
        axes[0].legend(*scatter.legend_elements(), title="True Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].set_title(title_true + " (Spectral Embedding)")
        axes[0].set_xlabel("Spectral Component 1")
        axes[0].set_ylabel("Spectral Component 2")
    else:
        axes[0].scatter(U[:, 0], U[:, 1], s=10, color='grey')
        axes[0].set_title("No True Labels Available")
        axes[0].set_xlabel("Spectral Component 1")
        axes[0].set_ylabel("Spectral Component 2")
    
    # Final Clusters
    scatter = axes[1].scatter(U[:, 0], U[:, 1], c=final_labels, cmap='tab10', s=10)
    axes[1].legend(*scatter.legend_elements(), title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].set_title(title_final + " (Spectral Embedding)")
    axes[1].set_xlabel("Spectral Component 1")
    axes[1].set_ylabel("Spectral Component 2")
    
    plt.tight_layout()
    plt.show()

def plot_final_clustering_pca(X_pca, y, final_labels, title_true="True Labels", title_final="Final Clustering Results"):
    """
    Plot a side-by-side comparison of true labels vs. determined clusters using PCA-reduced data.
    
    Parameters:
    - X_pca: PCA-reduced data of shape (n_samples, 2)
    - y: True labels (array-like) or None
    - final_labels: Cluster labels determined by FSEC
    - title_true: Title for the true labels plot
    - title_final: Title for the final clustering plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot True Labels
    if y is not None:
        scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=10)
        legend1 = axes[0].legend(*scatter.legend_elements(), title="True Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].add_artist(legend1)
        axes[0].set_title(f"{title_true} (PCA Reduced)")
        axes[0].set_xlabel("PCA Component 1")
        axes[0].set_ylabel("PCA Component 2")
    else:
        axes[0].scatter(X_pca[:, 0], X_pca[:, 1], s=10, color='grey')
        axes[0].set_title("No True Labels Available")
        axes[0].set_xlabel("PCA Component 1")
        axes[0].set_ylabel("PCA Component 2")
    
    # Plot Final Clusters
    scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=final_labels, cmap='tab10', s=10)
    legend2 = axes[1].legend(*scatter.legend_elements(), title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].add_artist(legend2)
    axes[1].set_title(f"{title_final} (PCA Reduced)")
    axes[1].set_xlabel("PCA Component 1")
    axes[1].set_ylabel("PCA Component 2")
    
    plt.tight_layout()
    plt.show()

