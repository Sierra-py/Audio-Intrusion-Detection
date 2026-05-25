""" 
Check for each category in the raw data folder if the samples are distributed in clusters, and check if how different they are.
"""
"""Issues : 
The scripts pauses for each folder. And the charts that are getting saved are overwritten because it uses same name  for everything.
The plot also don't have a title.
"""

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from config.config import SR, N_MELS

def extract_features(filepath):
    try:
        y, _ = librosa.load(filepath, sr=SR, mono=True, duration=10)
        
        # Mel spectrogram summary
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Feature vector: stats over time axis
        features = np.concatenate([
            np.mean(mel_db, axis=1),
            np.std(mel_db, axis=1),
            np.mean(librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20), axis=1),
            np.std(librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20), axis=1),
            [librosa.feature.spectral_centroid(y=y, sr=SR).mean()],
            [librosa.feature.spectral_rolloff(y=y, sr=SR).mean()],
            [librosa.feature.zero_crossing_rate(y).mean()]
        ])
        return features
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def load_folder(folder_path):
    features, filenames = [], []
    files = [f for f in os.listdir(folder_path) if f.endswith(('.mp3', '.wav', '.ogg'))]
    
    for i, fname in enumerate(files):
        print(f"Processing {i+1}/{len(files)}: {fname}", end='\r')
        feat = extract_features(os.path.join(folder_path, fname))
        if feat is not None:
            features.append(feat)
            filenames.append(fname)
    
    return np.array(features), filenames

def find_optimal_clusters(features_scaled, max_k=8):
    inertias, silhouettes = [], []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(features_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(features_scaled, labels))
    
    # Plot elbow + silhouette
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    
    ax2.plot(k_range, silhouettes, 'ro-')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score (higher = better)')
    
    plt.tight_layout()
    plt.savefig('cluster_analysis.png', dpi=150)
    plt.show()
    
    best_k = k_range[np.argmax(silhouettes)]
    print(f"\nOptimal k by silhouette: {best_k}")
    return best_k

def cluster_and_visualize(features_scaled, filenames, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(features_scaled)
    
    # PCA to 2D for visualization
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features_scaled)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'Audio Clusters (k={k})')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.savefig('clusters_pca.png', dpi=150)
    plt.show()
    
    # Print cluster contents
    for cluster_id in range(k):
        cluster_files = [filenames[i] for i, l in enumerate(labels) if l == cluster_id]
        print(f"\nCluster {cluster_id}: {len(cluster_files)} files")
        for f in cluster_files[:5]:  # show first 5
            print(f"  {f}")
        if len(cluster_files) > 5:
            print(f"  ... and {len(cluster_files) - 5} more")
    
    return labels

if __name__ == "__main__":
    for category in os.listdir("data/raw"):
        folder = f"data/raw/{category}"  # change this
        print(f"\n\n\033[91mProcessing {category}\033[0m\n\n")

        print("Extracting features...")
        features, filenames = load_folder(folder)
        print(f"\nLoaded {len(features)} files")
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        print("\nFinding optimal clusters...")
        best_k = find_optimal_clusters(features_scaled)
        
        print(f"\nClustering with k={best_k}...")
        labels = cluster_and_visualize(features_scaled, filenames, best_k)