from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def kmeans_clustering(X, n_clusters=3, random_state=42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    model.fit(X)
    return model

def pca_transform(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return pca, X_reduced
