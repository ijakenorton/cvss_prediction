import gensim
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
import nltk
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import utils
import os

nltk.download("stopwords", quiet=True)
version = 31


def run():
    nvd_data = utils.read_data(f"../data/nvd_{version}_cleaned.pkl")
    data = nvd_data["data"]
    v3_dimensions = utils.v3_dimensions
    for metric in v3_dimensions:
        for value in v3_dimensions[metric].values():
            descriptions = utils.filter_by_metric_score(data, metric, value)
            descriptions = list(map(lambda x: x["description"][0], descriptions))
            train(descriptions, metric, value)


def train(descriptions, metric, value):
    output_dir = "word2vec_output"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{output_dir}/word2vec_output_{metric}_{value}.txt"

    custom_stopwords = set(stopwords.words("english")).union(set(STOPWORDS))
    custom_stopwords.update(
        ["vulnerability", "via", "allows", "attacker", "could", "lead"]
    )

    def preprocess(text):
        return [
            word
            for word in gensim.utils.simple_preprocess(text)
            if word not in custom_stopwords and len(word) > 2
        ]

    texts = [preprocess(text) for text in descriptions]
    texts = [text for text in texts if text]

    with open(file_name, "w") as f:
        f.write(f"Analysis for {metric} with value {value}\n\n")
        f.write(f"Number of documents: {len(texts)}\n")
        f.write(f"Total number of words: {sum(len(text) for text in texts)}\n\n")

    # Train Word2Vec model
    model = Word2Vec(sentences=texts, vector_size=100, window=5, min_count=5, workers=4)

    # Get vocabulary
    vocab = list(model.wv.key_to_index.keys())

    # Get word vectors
    word_vectors = model.wv[vocab]

    # Perform K-means clustering
    num_clusters = 10  # You can adjust this
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(word_vectors)

    # Get top words for each cluster
    def get_top_words(cluster_id, n_words=10):
        words = [vocab[i] for i, c in enumerate(kmeans.labels_) if c == cluster_id]
        return sorted(
            words, key=lambda w: model.wv.get_vecattr(w, "count"), reverse=True
        )[:n_words]

    # Write results
    with open(file_name, "a") as f:
        f.write("Word2Vec Topics:\n")
        for i in range(num_clusters):
            top_words = get_top_words(i)
            f.write(f"Topic {i}: {', '.join(top_words)}\n")

    # Visualize clusters (2D projection using PCA)
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    two_d = pca.fit_transform(word_vectors)

    plt.figure(figsize=(12, 8))
    plt.scatter(two_d[:, 0], two_d[:, 1], c=kmeans.labels_, cmap="viridis")
    plt.title("Word Clusters")
    plt.savefig(f"{output_dir}/word_clusters_{metric}_{value}.png")
    plt.close()

    # Save model
    model_file = f"{output_dir}/word2vec_model_{metric}_{value}.model"
    model.save(model_file)

    with open(file_name, "a") as f:
        f.write(f"\nWord2Vec model saved. You can load it later using:\n")
        f.write(f"model = Word2Vec.load('{model_file}')\n")


def main():
    run()


if __name__ == "__main__":
    main()
