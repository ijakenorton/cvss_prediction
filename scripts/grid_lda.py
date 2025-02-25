import gensim
from gensim import corpora
from gensim.models import LdaModel, LdaMulticore, Word2Vec, KeyedVectors, CoherenceModel
from gensim.models.fasttext import load_facebook_vectors
from gensim.parsing.preprocessing import STOPWORDS
import gensim.downloader as api
import nltk
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cvss_types import Metrics_t, MetricNames
import utils
import os
import json
from itertools import product
from tqdm import tqdm
from collections import defaultdict
import random

nltk.download("stopwords", quiet=True)
version = 31


def run_all():
    import config

    nvd_data = utils.read_data(f"../data/nvd_{version}_cleaned.pkl")
    data = nvd_data["data"]
    # data = balance_data_by_integrity_impact(data)
    data = balance_data_by_metric(data, config.current_metric)
    descriptions = list(map(lambda x: x["description"][0], data))
    metric_values = list(map(lambda x: x["cvssData"], data))
    train(descriptions, "all_metrics", "all_values", metric_values)


def balance_data_by_metric(data, metric: Metrics_t, max_per_category=20000):
    # Group data by metric
    grouped_data = defaultdict(list)
    for item in data:
        integrity_impact = item["cvssData"].get(metric, "UNKNOWN")
        grouped_data[integrity_impact].append(item)

    # Balance the data
    balanced_data = []
    for impact, items in grouped_data.items():
        balanced_data.extend(random.sample(items, min(len(items), max_per_category)))

    # Shuffle the balanced data
    random.shuffle(balanced_data)

    return balanced_data


def train(descriptions, metric, value, metric_values):
    import config

    output_dir = f"lda_word2vec_grid_search_{config.current_metric}_{config.num_topics}"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{output_dir}/lda_word2vec_balanced_{metric}_{value}.txt"

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

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    corpus = [dictionary.doc2bow(text) for text in texts]

    with open(file_name, "w") as f:
        f.write(f"Analysis for {metric} with value {value}\n\n")
        f.write(f"Number of documents in corpus: {len(corpus)}\n")
        f.write(f"Number of unique tokens: {len(dictionary)}\n")
        f.write(
            f"Total number of terms in corpus: {sum(count for doc in corpus for _, count in doc)}\n\n"
        )

    # Train Word2Vec model
    w2v_model = Word2Vec(
        sentences=texts, vector_size=100, window=5, min_count=2, workers=4
    )

    def compute_coherence_word2vec(topic_words, w2v_model):
        if len(topic_words) < 2:
            return 0
        score = 0
        for i in range(len(topic_words)):
            for j in range(i + 1, len(topic_words)):
                if topic_words[i] in w2v_model.wv and topic_words[j] in w2v_model.wv:
                    score += w2v_model.wv.similarity(topic_words[i], topic_words[j])
        return score / (len(topic_words) * (len(topic_words) - 1) / 2)

    def run_lda_model(
        corpus,
        dictionary,
        num_topics,
        alpha,
        eta,
        passes,
        iterations,
        random_state,
        chunk_size,
        update_every,
    ):
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            alpha=alpha,
            eta=eta,
            random_state=random_state,
            chunksize=chunk_size,
            passes=passes,
            iterations=iterations,
            update_every=update_every,  # Determines online learning. Set to 1 for pure online learning.
            eval_every=None,  # Disable perplexity evaluation for speed
        )

        # Calculate custom Word2Vec coherence
        w2v_coherence = np.mean(
            [
                compute_coherence_word2vec(
                    [word for word, _ in model.show_topic(topic_id, topn=10)],
                    w2v_model,
                )
                for topic_id in range(model.num_topics)
            ]
        )

        # Calculate C_v coherence
        coherence_model_cv = CoherenceModel(
            model=model, texts=texts, dictionary=dictionary, coherence="c_v"
        )
        cv_coherence = coherence_model_cv.get_coherence()

        # Calculate perplexity
        perplexity = np.exp(-model.log_perplexity(corpus))

        return model, w2v_coherence, cv_coherence, perplexity

    # Grid search parameters
    num_topics_range = range(config.num_topics, config.num_topics + 1, 1)
    alpha_range = ["symmetric", "asymmetric", "auto"]  # Try different alpha settings
    eta_range = [None, "auto", 0.1, 0.01]  # Try different eta settings
    passes_range = [1, 2, 5]  # Fewer passes for online learning
    chunk_size_range = [2000, 4000, 8000]  # Try different chunk sizes
    update_every_range = [1, 2]  # 1 for pure online, 2 for mini-batch
    iterations_range = [50, 100, 200]
    random_state_range = [0]
    total_iterations = (
        len(num_topics_range)
        * len(alpha_range)
        * len(eta_range)
        * len(passes_range)
        * len(iterations_range)
        * len(random_state_range)
        * len(chunk_size_range)
        * len(update_every_range)
    )
    pbar = tqdm(total=total_iterations, desc="Grid search progress")
    # Run grid search
    results = []
    for (
        num_topics,
        alpha,
        eta,
        passes,
        chunk_size,
        update_every,
        iterations,
        random_state,
    ) in product(
        num_topics_range,
        alpha_range,
        eta_range,
        passes_range,
        chunk_size_range,
        update_every_range,
        iterations_range,
        random_state_range,
    ):
        model, w2v_coherence, cv_coherence, perplexity = run_lda_model(
            corpus,
            dictionary,
            num_topics,
            alpha,
            eta,
            passes,
            iterations,
            random_state,
            chunk_size,
            update_every,
        )
        results.append(
            {
                "num_topics": num_topics,
                "alpha": alpha,
                "eta": eta,
                "passes": passes,
                "chunk_size": chunk_size,
                "update_every": update_every,
                "iterations": iterations,
                "w2v_coherence": w2v_coherence,
                "cv_coherence": cv_coherence,
                "perplexity": perplexity,
                "model": model,
                "seed": random_state,
            }
        )
        pbar.update(1)

    # pbar.close()
    # Sort results by coherence score
    results.sort(key=lambda x: x["cv_coherence"], reverse=True)

    # Write grid search results
    with open(file_name, "a") as f:
        f.write("\nGrid Search Results:\n")
        for result in results:
            f.write(
                f"Num Topics: {result['num_topics']}, Alpha: {result['alpha']}, Eta: {result['eta']}, "
                f"Passes: {result['passes']}, Chunk Size: {result['chunk_size']}, "
                f"Update Every: {result['update_every']}, Iterations: {result['iterations']}, "
                f"Seed: {result['seed']}, "
                f"Word2Vec Coherence: {result['w2v_coherence']:.4f}, "
                f"C_v Coherence: {result['cv_coherence']:.4f}, "
                f"Perplexity: {result['perplexity']:.2f}\n"
            )

    # Save top 5 models
    for i, result in enumerate(results[:5]):
        model = result["model"]
        model_name = f"lda_model_t{result['num_topics']}_a{result['alpha']}_e{result['eta']}_p{result['passes']}_i{result['iterations']}_seed{result['seed']}"
        model_file = f"{output_dir}/{model_name}.gensim"
        model.save(model_file)

        # Get topic assignments for each document
        doc_topics = [model.get_document_topics(doc) for doc in corpus]
        topic_assignments = [
            max(topics, key=lambda x: x[1])[0] for topics in doc_topics
        ]

        results = []
        for j, (description, topic, metric_value) in enumerate(
            zip(descriptions, topic_assignments, metric_values)
        ):
            results.append(
                {
                    "document_index": j,
                    "description": description,
                    "topic": topic,
                    "cvssData": metric_value,
                }
            )

        # Save results as JSON
        with open(f"{output_dir}/topic_assignments_{model_name}.json", "w") as f:
            json.dump(results, f, indent=2)

        # Print and save top words for each topic
        with open(file_name, "a") as f:
            f.write(f"\nTop 50 words for each topic (Model: {model_name}):\n")
            for idx, topic in model.print_topics(-1, num_words=50):
                f.write(f"Topic {idx}:\n")
                words = [word for word, _ in model.show_topic(idx, topn=50)]
                for word in words:
                    f.write(f"  - {word}\n")
                f.write("\n")

        with open(file_name, "a") as f:
            print(
                f"\nTopic assignments saved to: {output_dir}/topic_assignments_{model_name}.json\n"
            )
            f.write(
                f"\nTopic assignments saved to: {output_dir}/topic_assignments_{model_name}.json\n"
            )
            f.write(f"Model saved to: {model_file}\n")

    dict_file = f"{output_dir}/dictionary.gensim"
    w2v_model_file = f"{output_dir}/word2vec_model.model"
    dictionary.save(dict_file)
    w2v_model.save(w2v_model_file)

    with open(file_name, "a") as f:
        f.write(f"\nDictionary saved to: {dict_file}\n")
        f.write(f"Word2Vec model saved to: {w2v_model_file}\n")


def main():
    run_all()


if __name__ == "__main__":
    main()
