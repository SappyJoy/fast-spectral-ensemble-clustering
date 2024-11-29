from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from fsec.clustering import FSEC
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = make_blobs(n_samples=1000, centers=5, random_state=42)
X = StandardScaler().fit_transform(X)

# Apply FSEC
fsec = FSEC(final_n_clusters=5)
labels = fsec.fit_predict(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10)
plt.title("FSEC Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

