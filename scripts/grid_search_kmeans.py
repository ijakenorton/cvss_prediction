import gensim
from gensim import corpora
from gensim.models import FastText, KeyedVectors
from gensim.parsing.preprocessing import STOPWORDS
import gensim.downloader as api
import nltk
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils
import os
import json
from itertools import product
from tqdm import tqdm
from collections import defaultdict
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

nltk.download("stopwords", quiet=True)
version = 31


def run_all():
    nvd_data = utils.read_data(f"../data/nvd_{version}_cleaned.pkl")
    data = nvd_data["data"]
    data = balance_data_by_integrity_impact(data)
    descriptions = list(map(lambda x: x["description"][0], data))
    metric_values = list(map(lambda x: x["cvssData"], data))
    train(descriptions, "all_metrics", "all_values", metric_values)


def balance_data_by_integrity_impact(data, max_per_category=20000):
    # Group data by integrity impact
    grouped_data = defaultdict(list)
    for item in data:
        integrity_impact = item["cvssData"].get("integrityImpact", "UNKNOWN")
        grouped_data[integrity_impact].append(item)

    # Balance the data
    balanced_data = []
    for impact, items in grouped_data.items():
        balanced_data.extend(random.sample(items, min(len(items), max_per_category)))

    # Shuffle the balanced data
    random.shuffle(balanced_data)

    return balanced_data


def train(descriptions, metric, value, metric_values):
    num_clusters = 100
    output_dir = f"kmeans_fasttext_{num_clusters}"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{output_dir}/kmeans_fasttext_{metric}_{value}.txt"

    custom_stopwords = set(stopwords.words("english")).union(set(STOPWORDS))
    custom_stopwords.update(
        ["vulnerability", "via", "allows", "attacker", "could", "lead", "leads"]
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
        f.write(f"Number of documents: {len(texts)}\n\n")

    # Load FastText vectors
    fasttext_vectors_path = os.path.join(os.getcwd(), "fasttext_vectors.kv")

    if not os.path.exists(fasttext_vectors_path):
        print("Downloading FastText vectors. This may take a while...")
        fasttext_vectors = api.load("fasttext-wiki-news-subwords-300")
        fasttext_vectors.save(fasttext_vectors_path)
    else:
        print("Loading existing FastText vectors...")
        fasttext_vectors = KeyedVectors.load(fasttext_vectors_path)

    # Create document vectors
    def get_doc_vector(doc):
        vec = np.zeros(fasttext_vectors.vector_size)
        count = 0
        for word in doc:
            if word in fasttext_vectors:
                vec += fasttext_vectors[word]
                count += 1
        return vec / count if count > 0 else vec

    doc_vectors = np.array([get_doc_vector(doc) for doc in texts])

    # Run K-means clustering
    def run_kmeans(n_clusters, random_state):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(doc_vectors)
        silhouette_avg = silhouette_score(doc_vectors, cluster_labels)
        return kmeans, cluster_labels, silhouette_avg

    # Grid search parameters
    n_clusters_range = range(num_clusters, num_clusters + 1, 1)
    random_state_range = [0]
    total_iterations = len(n_clusters_range) * len(random_state_range)
    pbar = tqdm(total=total_iterations, desc="Grid search progress")

    # Run grid search
    results = []
    for n_clusters, random_state in product(n_clusters_range, random_state_range):
        kmeans, cluster_labels, silhouette_avg = run_kmeans(n_clusters, random_state)
        results.append(
            {
                "n_clusters": n_clusters,
                "random_state": random_state,
                "silhouette_score": silhouette_avg,
                "kmeans": kmeans,
                "cluster_labels": cluster_labels,
            }
        )
        pbar.update(1)

    pbar.close()

    # Sort results by silhouette score
    results.sort(key=lambda x: x["silhouette_score"], reverse=True)

    # Write grid search results
    with open(file_name, "a") as f:
        f.write("\nGrid Search Results:\n")
        for result in results:
            f.write(
                f"Num Clusters: {result['n_clusters']}, "
                f"Random State: {result['random_state']}, "
                f"Silhouette Score: {result['silhouette_score']}\n"
            )

    # Save top 5 models
    for i, result in enumerate(results[:5]):
        kmeans = result["kmeans"]
        cluster_labels = result["cluster_labels"]
        model_name = (
            f"kmeans_model_c{result['n_clusters']}_seed{result['random_state']}"
        )
        model_file = f"{output_dir}/{model_name}.pkl"

        # Save K-means model
        import joblib

        joblib.dump(kmeans, model_file)

        # Save cluster assignments
        cluster_assignments = []
        for j, (description, cluster, metric_value) in enumerate(
            zip(descriptions, cluster_labels, metric_values)
        ):
            cluster_assignments.append(
                {
                    "document_index": j,
                    "description": description,
                    "cluster": int(cluster),
                    "cvssData": metric_value,
                }
            )

        # Save results as JSON
        with open(f"{output_dir}/cluster_assignments_{model_name}.json", "w") as f:
            json.dump(cluster_assignments, f, indent=2)

        # Print and save top words for each cluster
        with open(file_name, "a") as f:
            f.write(f"\nTop 50 words for each cluster (Model: {model_name}):\n")
            for cluster_id in range(kmeans.n_clusters):
                cluster_docs = [
                    doc
                    for doc, label in zip(texts, cluster_labels)
                    if label == cluster_id
                ]
                cluster_words = [word for doc in cluster_docs for word in doc]
                word_freq = defaultdict(int)
                for word in cluster_words:
                    word_freq[word] += 1
                top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[
                    :50
                ]

                f.write(f"Cluster {cluster_id}:\n")
                for word, freq in top_words:
                    f.write(f"  - {word} ({freq})\n")
                f.write("\n")

        with open(file_name, "a") as f:
            print(
                f"\nCluster assignments saved to: {output_dir}/cluster_assignments_{model_name}.json\n"
            )
            f.write(
                f"\nCluster assignments saved to: {output_dir}/cluster_assignments_{model_name}.json\n"
            )
            f.write(f"Model saved to: {model_file}\n")

    fasttext_model_file = f"{output_dir}/fasttext_vectors.kv"
    fasttext_vectors.save(fasttext_model_file)

    with open(file_name, "a") as f:
        f.write(f"\nFastText vectors saved to: {fasttext_model_file}\n")


def main():
    run_all()


if __name__ == "__main__":
    main()
