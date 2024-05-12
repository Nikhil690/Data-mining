import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = np.array([[1, 1], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(data)
centroids = kmeans.cluster_centers_

# Get assigned cluster labels
labels = kmeans.labels_

print("Cluster centroids:")
print(kmeans.cluster_centers_)

print("Assigned cluster labels:")
print(kmeans.labels_)


parameters = {
    'n_clusters': [2, 3, 4],  # Vary the number of clusters
    'init': ['k-means++', 'random'],  # Experiment with different initialization methods
    'max_iter': [100, 200],  # Adjust the maximum number of iterations
    'tol': [1e-4, 1e-3]  # Change the tolerance for convergence
}

# Loop over different parameter combinations
for n_clusters in parameters['n_clusters']:
    for init in parameters['init']:
        for max_iter in parameters['max_iter']:
            for tol in parameters['tol']:
                # Create KMeans instance with current parameter values
                kmeans = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, tol=tol, random_state=0)
                
                # Fit the KMeans model to the data
                kmeans.fit(data)
                
                # Calculate the inertia (within-cluster sum of squared distances)
                inertia = kmeans.inertia_
                
                # Calculate MSE (mean squared error)
                num_samples = data.shape[0]
                mse = inertia / num_samples
                
                # Print parameter values and MSE
                print(f"Parameters: n_clusters={n_clusters}, init='{init}', max_iter={max_iter}, tol={tol}")
                print(f"MSE: {mse}")
                print(f"Cluster centroids:\n{kmeans.cluster_centers_}")
                print(f"Assigned cluster labels:\n{kmeans.labels_}")
                print("\n")
                
                # Plot MSE after each iteration
                iterations = range(1, max_iter + 1)
                mse_history = []
                for i in range(1, max_iter + 1):
                    kmeans_iter = KMeans(n_clusters=n_clusters, init=init, max_iter=i, tol=tol, random_state=0)
                    kmeans_iter.fit(data)
                    inertia_iter = kmeans_iter.inertia_
                    mse_iter = inertia_iter / num_samples
                    mse_history.append(mse_iter)
                plt.plot(iterations, mse_history, marker='o')
                plt.title('MSE vs. Iterations')
                plt.xlabel('Iterations')
                plt.ylabel('Mean Squared Error (MSE)')
                plt.grid(True)
                plt.show()
