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

import utils

true_k = 8
version = 31
nvd_data = utils.read_data(f"../data/nvd_{version}_cleaned.pkl")

data = nvd_data["data"]
descriptions = list(map(lambda x: x["description"][0], data))
nltk.download("stopwords", quiet=True)

custom_stopwords = set(stopwords.words("english")).union(set(STOPWORDS))
custom_stopwords.update(["vulnerability", "via", "allows", "attacker", "could", "lead"])


def preprocess(text):
    return [
        word
        for word in gensim.utils.simple_preprocess(text)
        if word not in custom_stopwords and len(word) > 2
    ]


# Use a smaller subset of the data for quicker iterations


texts = [preprocess(text) for text in descriptions]
texts = [text for text in texts if text]


dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=2, no_above=0.9)

corpus = [dictionary.doc2bow(text) for text in texts]

print(f"Number of documents in corpus: {len(corpus)}")
print(f"Number of unique tokens: {len(dictionary)}")
print(
    f"Total number of terms in corpus: {sum(count for doc in corpus for _, count in doc)}"
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


# Compute coherence values for different numbers of topics
model_list, coherence_values = compute_coherence_values(
    corpus=corpus, dictionary=dictionary, texts=texts, start=2, limit=40, step=6
)

# Plot coherence scores
limit = 40
start = 2
step = 6
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc="best")
plt.show()

# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print(f"Num Topics: {m}, Coherence Score: {cv}")

# Select the model with the highest coherence score
optimal_model = model_list[coherence_values.index(max(coherence_values))]
print(f"\nOptimal number of topics: {optimal_model.num_topics}")

# Print the top words for each topic in the optimal model
print("\nTop 10 words for each topic in the optimal model:")
for idx, topic in optimal_model.print_topics(-1):
    print(f"Topic: {idx} \nWords: {topic}\n")


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
import pandas as pd

df_dominant_topic = pd.DataFrame(
    df_topic_sents_keywords,
    columns=["Dominant_Topic", "Perc_Contribution", "Topic_Keywords"],
)

# Show
print(df_dominant_topic.head(10))

# Compute topic distribution for all documents
doc_topics = [optimal_model.get_document_topics(doc) for doc in corpus]


def topic_distribution(doc_topics, num_topics):
    return np.array([dict(doc_topics).get(i, 0.0) for i in range(num_topics)])


X_topics = np.array(
    [topic_distribution(doc, optimal_model.num_topics) for doc in doc_topics]
)

print("\nShape of topic distribution matrix:", X_topics.shape)
print("Sample topic distribution for a document:\n", X_topics[0])

# Save the model for future use
optimal_model.save("lda_model.gensim")
dictionary.save("dictionary.gensim")

print("\nModel and dictionary saved. You can load them later using:")
print("lda_model = LdaModel.load('lda_model.gensim')")
print("dictionary = corpora.Dictionary.load('dictionary.gensim')")
