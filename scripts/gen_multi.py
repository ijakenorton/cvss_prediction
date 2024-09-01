import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models import LdaMulticore, Phrases
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS
from time import time
import nltk
from nltk.corpus import stopwords
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils
import os
from pprint import pprint

nltk.download("stopwords", quiet=True)
true_k = 8
version = 31


def run():
    nvd_data = utils.read_data(f"../data/nvd_{version}_cleaned.pkl")
    data = nvd_data["data"]
    v3_dimensions = utils.v3_dimensions
    for metric in v3_dimensions:
        for value in v3_dimensions[metric].values():
            descriptions = utils.filter_by_metric_score(data, metric, value)
            descriptions = list(map(lambda x: x["description"][0], descriptions))
            pprint((metric, value))
            train(descriptions, metric, value)


def train(descriptions, metric, value):
    # Create a directory for output if it doesn't exist
    output_dir = "lda_output"
    os.makedirs(output_dir, exist_ok=True)

    # Create a file name based on metric and value
    file_name = f"{output_dir}/lda_output_{metric}_{value}.txt"

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

    def compute_coherence_values(corpus, dictionary, texts, limit, start=2, step=3):
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
            coherencemodel = CoherenceModel(
                model=model, texts=texts, dictionary=dictionary, coherence="c_v"
            )
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values

    limit = 20
    start = 2
    step = 2

    # Compute coherence values for different numbers of topics
    model_list, coherence_values = compute_coherence_values(
        corpus=corpus,
        dictionary=dictionary,
        texts=texts,
        start=start,
        limit=limit,
        step=step,
    )

    # Plot coherence scores
    x = range(start, limit, step)
    plt.figure()
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc="best")
    plt.savefig(f"{output_dir}/coherence_plot_{metric}_{value}.png")
    plt.close()

    # Write the coherence scores
    with open(file_name, "a") as f:
        f.write("Coherence Scores:\n")
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

    # Function to format topics
    def format_topics_sentences(ldamodel, corpus, texts):
        sent_topics_df = []
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # Consider only the dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df.append(
                        [int(topic_num), round(prop_topic, 4), topic_keywords]
                    )
                else:
                    break
        return sent_topics_df

    # Format the results
    df_topic_sents_keywords = format_topics_sentences(
        ldamodel=optimal_model, corpus=corpus, texts=texts
    )

    # Convert to DataFrame
    df_dominant_topic = pd.DataFrame(
        df_topic_sents_keywords,
        columns=["Dominant_Topic", "Perc_Contribution", "Topic_Keywords"],
    )

    # Write DataFrame to file
    with open(file_name, "a") as f:
        f.write("\nDominant Topics:\n")
        f.write(df_dominant_topic.head(50).to_string())
        f.write("\n\n")

    # Compute topic distribution for all documents
    doc_topics = [optimal_model.get_document_topics(doc) for doc in corpus]

    def topic_distribution(doc_topics, num_topics):
        return np.array([dict(doc_topics).get(i, 0.0) for i in range(num_topics)])

    X_topics = np.array(
        [topic_distribution(doc, optimal_model.num_topics) for doc in doc_topics]
    )

    with open(file_name, "a") as f:
        f.write(f"\nShape of topic distribution matrix: {X_topics.shape}\n")
        f.write(f"Sample topic distribution for a document:\n{X_topics[0]}\n")

    # Save the model for future use
    model_file = f"{output_dir}/lda_model_{metric}_{value}.gensim"
    dict_file = f"{output_dir}/dictionary_{metric}_{value}.gensim"
    optimal_model.save(model_file)
    dictionary.save(dict_file)

    with open(file_name, "a") as f:
        f.write("\nModel and dictionary saved. You can load them later using:\n")
        f.write(f"lda_model = LdaModel.load('{model_file}')\n")
        f.write(f"dictionary = corpora.Dictionary.load('{dict_file}')\n")


def main():
    run()


if __name__ == "__main__":
    main()
