# Text Clustering
from sentence_transformer import load_model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt


model = load_model()


def text_clustering(news_documents):
    # Convert all news documents into embedding vectors.
    news_embeddings = model.encode(news_documents)

    # Create a K-Means model with three clusters.
    kmeans = KMeans(
          n_clusters=4,
          random_state=42,
          n_init=10
      )

    # Assign each document to a cluster.
    cluster_labels = kmeans.fit_predict(news_embeddings)

    # Display the documents grouped by their predicted cluster.
    clustered_news = pd.DataFrame({
        "document": news_documents,
        "cluster": cluster_labels
    })

    print(clustered_news.sort_values("cluster").to_string(index=False))

    return news_embeddings, cluster_labels


'''
Since Sentence Transformer embeddings are 384-dimensional, 
we cannot directly plot them in 2D. 
A common approach is to reduce the embeddings to two dimensions with PCA, 
then plot the four K-Means clusters.
'''


def plot_clusters(news_embeddings, cluster_labels):
    # Reduce the embedding vectors from 384 dimensions to 2 dimensions.
    pca = PCA(n_components=2, random_state=42)
    reduced_embeddings = pca.fit_transform(news_embeddings)

    # Create the scatter plot.
    plt.figure(figsize=(10, 7))

    for cluster in sorted(set(cluster_labels)):
        points = reduced_embeddings[cluster_labels == cluster]

        plt.scatter(
            points[:, 0],
            points[:, 1],
            label=f'Cluster {cluster}',
            alpha=0.7
        )

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('News Document Clusters')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    # Read one news document from each line.
    with open('documents/news_documents.txt', 'r', encoding='utf-8') as file:
        news_documents = [line.strip() for line in file if line.strip()]

    print(f'Loaded {len(news_documents)} documents.')

    news_embeddings, cluster_labels = text_clustering(news_documents)

    # Plot clusters
    plot_clusters(news_embeddings, cluster_labels)


if __name__ == '__main__':
    main()
