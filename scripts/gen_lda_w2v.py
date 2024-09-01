import gensim
from gensim import corpora
from gensim.models import LdaMulticore, Word2Vec
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.corpus import stopwords
import numpy as np
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
    output_dir = "lda_word2vec_output"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{output_dir}/lda_word2vec_output_{metric}_{value}.txt"

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

    def compute_coherence_values(
        corpus, dictionary, texts, w2v_model, limit, start=2, step=3
    ):
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = LdaMulticore(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=100,
                chunksize=2000,
                passes=20,
                iterations=400,
                workers=6,
            )
            model_list.append(model)

            topics = model.print_topics(num_words=10)
            coherence = np.mean(
                [
                    compute_coherence_word2vec(
                        [word for word, _ in model.show_topic(topic_id, topn=10)],
                        w2v_model,
                    )
                    for topic_id in range(model.num_topics)
                ]
            )
            coherence_values.append(coherence)

        return model_list, coherence_values

    limit = 20
    start = 2
    step = 2

    # Compute coherence values for different numbers of topics
    model_list, coherence_values = compute_coherence_values(
        corpus=corpus,
        dictionary=dictionary,
        texts=texts,
        w2v_model=w2v_model,
        start=start,
        limit=limit,
        step=step,
    )

    # Plot coherence scores
    x = range(start, limit, step)
    plt.figure(figsize=(12, 8))
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Word2Vec-based Coherence Score")
    plt.title(f"Topic Coherence Scores ({metric}: {value})")
    plt.savefig(f"{output_dir}/coherence_plot_{metric}_{value}.png")
    plt.close()

    # Write the coherence scores
    with open(file_name, "a") as f:
        f.write("Word2Vec-based Coherence Scores:\n")
        for m, cv in zip(x, coherence_values):
            f.write(f"Num Topics: {m}, Coherence Score: {cv}\n")

    # Select the model with the highest coherence score
    optimal_model = model_list[coherence_values.index(max(coherence_values))]

    with open(file_name, "a") as f:
        f.write(f"\nOptimal number of topics: {optimal_model.num_topics}\n")

        # Write the top words for each topic in the optimal model
        f.write("\nTop 10 words for each topic in the optimal model:\n")
        for idx, topic in optimal_model.print_topics(-1):
            f.write(f"Topic: {idx} \nWords: {topic}\n\n")

    # Save the models
    lda_model_file = f"{output_dir}/lda_model_{metric}_{value}.gensim"
    dict_file = f"{output_dir}/dictionary_{metric}_{value}.gensim"
    w2v_model_file = f"{output_dir}/word2vec_model_{metric}_{value}.model"
    optimal_model.save(lda_model_file)
    dictionary.save(dict_file)
    w2v_model.save(w2v_model_file)

    with open(file_name, "a") as f:
        f.write("\nModels saved. You can load them later using:\n")
        f.write(f"lda_model = LdaModel.load('{lda_model_file}')\n")
        f.write(f"dictionary = corpora.Dictionary.load('{dict_file}')\n")
        f.write(f"w2v_model = Word2Vec.load('{w2v_model_file}')\n")


def main():
    run()


if __name__ == "__main__":
    main()
