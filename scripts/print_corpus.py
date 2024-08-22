import gensim
from gensim import corpora
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.corpus import stopwords
from pprint import pprint
import utils


nltk.download("stopwords", quiet=True)
true_k = 8
version = 31
nvd_data = utils.read_data(f"../data/nvd_{version}_cleaned.pkl")
data = nvd_data["data"]
data = list(filter(lambda x: x["cvssData"]["attackVector"] == "LOCAL", data))
descriptions = list(map(lambda x: x["description"][0], data))

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
# texts = [text for text in texts if text]

# for i in range(10):
#     print(descriptions[i], "\n")
for i in range(50):
    print(texts[i], "\n")

# dictionary = corpora.Dictionary(texts)
# dictionary.filter_extremes(no_below=2, no_above=0.9)

# corpus = [dictionary.doc2bow(text) for text in texts]
