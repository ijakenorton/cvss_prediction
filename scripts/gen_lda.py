# import gensim


# Assuming 'descriptions' is your list of document texts
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models import LdaMulticore, Phrases
from gensim.parsing.preprocessing import STOPWORDS
from time import time
import nltk
from nltk.corpus import stopwords
import random


import utils

true_k = 8
version = 31
nvd_data = utils.read_data(f"../data/nvd_{version}_cleaned.pkl")

data = nvd_data["data"]
descriptions = list(map(lambda x: x["description"][0], data))
# Download NLTK stopwords
nltk.download("stopwords", quiet=True)

custom_stopwords = set(stopwords.words("english")).union(set(STOPWORDS))
custom_stopwords.update(["vulnerability", "via", "allows", "attacker", "could", "lead"])


def preprocess(text):
    return [
        word
        for word in gensim.utils.simple_preprocess(text)
        if word not in custom_stopwords and len(word) > 2
    ]


# Use a smaller subset of the data


texts = [preprocess(text) for text in descriptions]
texts = [text for text in texts if text]

# bigram = Phrases(texts, min_count=5)
# trigram = Phrases(bigram[texts])

# texts = [trigram[bigram[text]] for text in texts]

dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=2, no_above=0.9)

corpus = [dictionary.doc2bow(text) for text in texts]

print(f"Number of documents in corpus: {len(corpus)}")
print(f"Number of unique tokens: {len(dictionary)}")
print(
    f"Total number of terms in corpus: {sum(count for doc in corpus for _, count in doc)}"
)

num_topics = 8
chunksize = 2000
passes = 20
iterations = 400

print("Training LDA model...")
t0 = time()
try:
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=100,
        chunksize=chunksize,
        passes=passes,
        iterations=iterations,
        eval_every=None,
        workers=4,
    )
    print(f"LDA model training completed in {time() - t0:.3f} s")

    print("\nTop 10 words for each topic:")
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic: {idx} \nWords: {topic}\n")
except ValueError as e:
    print(f"Error during LDA model training: {e}")
    print("Corpus and dictionary details:")
    print(f"Dictionary: {dictionary}")
    print(f"First few corpus entries: {corpus[:5]}")

# Try to access the first topic
try:
    print("\nFirst topic:")
    print(lda_model.print_topic(0))
except:
    print("Unable to print first topic")
