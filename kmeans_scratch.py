import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import silhouette_samples
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from matplotlib import cm

class ScratchKMeans():
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, n_init=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
    
    # Problem 1: Initialization of Cluster Centers
    def initialize_centers(self, X):
        np.random.seed(None)  # Different initial values
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]
    
    # Problem 2: Compute Euclidean Distance
    def compute_distances(self, X, centers):
        return np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
    
    # Problem 3: Assign Clusters
    def assign_clusters(self, X, centers):
        distances = self.compute_distances(X, centers)
        return np.argmin(distances, axis=1)
    
    # Problem 4: Move Cluster Centers
    def move_centers(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
    
    # Problem 5: Iterative Optimization
    def fit(self, X):
        self.centers = self.initialize_centers(X)
        for _ in range(self.max_iter):
            labels = self.assign_clusters(X, self.centers)
            new_centers = self.move_centers(X, labels)
            if np.linalg.norm(self.centers - new_centers) < self.tol:
                break
            self.centers = new_centers
        self.labels_ = labels
    
    # Problem 6: Multiple Runs to Find Best Clusters
    def fit_best(self, X):
        best_sse = float('inf')
        best_centers = None
        best_labels = None
        for _ in range(self.n_init):
            self.fit(X)
            sse = np.sum((X - self.centers[self.labels_]) ** 2)
            if sse < best_sse:
                best_sse = sse
                best_centers = self.centers
                best_labels = self.labels_
        self.centers = best_centers
        self.labels_ = best_labels
    
    # Problem 7: Prediction
    def predict(self, X):
        return self.assign_clusters(X, self.centers)
    
    # Problem 8: Elbow Method
    def compute_sse(self, X, k_range):
        sse = []
        for k in k_range:
            self.n_clusters = k
            self.fit_best(X)
            sse.append(np.sum((X - self.centers[self.labels_]) ** 2))
        plt.plot(k_range, sse, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('SSE')
        plt.title('Elbow Method')
        plt.show()
    
    # Problem 9: Silhouette Plot
    def silhouette_plot(self, X):
        silhouette_vals = silhouette_samples(X, self.labels_)
        silhouette_avg = np.mean(silhouette_vals)
        y_ax_lower, y_ax_upper = 0, 0
        yticks = []
        plt.figure(figsize=(8, 6))
        for i, c in enumerate(np.unique(self.labels_)):
            c_silhouette_vals = silhouette_vals[self.labels_ == c]
            c_silhouette_vals.sort()
            y_ax_upper += len(c_silhouette_vals)
            color = cm.jet(float(i) / self.n_clusters)
            plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)
            yticks.append((y_ax_lower + y_ax_upper) / 2)
            y_ax_lower += len(c_silhouette_vals)
        plt.axvline(silhouette_avg, color="red", linestyle="--", label=f'Avg Silhouette Score: {silhouette_avg:.2f}')
        plt.yticks(yticks, [f'Cluster {i+1}' for i in range(self.n_clusters)])
        plt.ylabel('Cluster')
        plt.xlabel('Silhouette coefficient')
        plt.title('Silhouette Analysis')
        plt.legend()
        plt.show()
    
    # Problem 10: Selecting the Optimal k
    def select_optimal_k(self, X):
        print("Select k based on elbow method or silhouette score.")
        self.compute_sse(X, range(2, 10))
        self.silhouette_plot(X)
    
    # Problem 11: Comparison with Known Groups
    def compare_with_known_groups(self, X, true_labels):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=true_labels, palette='Set1', alpha=0.6, edgecolor='k')
        plt.title("Comparison of Clusters with Known Groups")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend(title="Known Groups")
        plt.show()
    
    # Problem 12: Useful Information for Wholesalers
    def cluster_summary(self, data, labels):
        df = pd.DataFrame(data)
        df['Cluster'] = labels
        summary = df.groupby('Cluster').mean()
        print("Cluster Summary Statistics:")
        print(summary)
        return summary

# Load the Wholesale customers dataset
dataset_path = '/content/Wholesale customers data.csv'
try:
    df = pd.read_csv(dataset_path)
    X = df.iloc[:, 2:].values  # Using relevant numerical columns
    
    # Perform PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Instantiate and fit the K-Means model
    kmeans = ScratchKMeans(n_clusters=3)
    kmeans.fit_best(X)
    
    # Print cluster centers
    print("Cluster Centers:", kmeans.centers)
    
    # Plot clustered data using PCA representation
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.6)
    plt.scatter(pca.transform(kmeans.centers)[:, 0], pca.transform(kmeans.centers)[:, 1], c='red', marker='X', s=200, label="Centers")
    plt.title("K-Means Clustering (PCA Reduced)")
    plt.legend()
    plt.show()
except FileNotFoundError:
    print(f"Error: Dataset file '{dataset_path}' not found. Please upload the file to the correct directory.")
