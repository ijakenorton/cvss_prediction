import numpy as np
from collections import defaultdict
from time import time
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import utils

version = 31


nvd_data = utils.read_data(f"../data/nvd_{version}_cleaned.pkl")

data = nvd_data["data"]
descriptions = list(map(lambda x: x["description"][0], data))
true_k = 8

evaluations = []
evaluations_std = []


def fit_and_evaluate(km, X, name=None, n_runs=5):
    name = km.__class__.__name__ if name is None else name

    train_times = []
    scores = defaultdict(list)
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        t0 = time()
        km.fit(X)
        train_times.append(time() - t0)
    train_times = np.asarray(train_times)

    print(f"clustering done in {train_times.mean():.2f} ± {train_times.std():.2f} s ")
    evaluation = {
        "estimator": name,
        "train_time": train_times.mean(),
    }
    evaluation_std = {
        "estimator": name,
        "train_time": train_times.std(),
    }
    for score_name, score_values in scores.items():
        mean_score, std_score = np.mean(score_values), np.std(score_values)
        print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")
        evaluation[score_name] = mean_score
        evaluation_std[score_name] = std_score
    evaluations.append(evaluation)
    evaluations_std.append(evaluation_std)


vectorizer = TfidfVectorizer(
    max_df=0.5,
    min_df=5,
    stop_words="english",
)
t0 = time()
X_tfidf = vectorizer.fit_transform(descriptions)


lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
t0 = time()
X_lsa = lsa.fit_transform(X_tfidf)
explained_variance = lsa[0].explained_variance_ratio_.sum()


minibatch_kmeans = MiniBatchKMeans(
    n_clusters=true_k,
    n_init=1,
    init_size=1000,
    batch_size=1000,
)

fit_and_evaluate(
    minibatch_kmeans,
    X_lsa,
    name="MiniBatchKMeans\nwith LSA on tf-idf vectors",
)

original_space_centroids = lsa[0].inverse_transform(minibatch_kmeans.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

for i in range(true_k):
    print(f"Cluster {i}: ", end="")
    for ind in order_centroids[i, :10]:
        print(f"{terms[ind]} ", end="")
    print()

# from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

# lsa_vectorizer = make_pipeline(
#     HashingVectorizer(stop_words="english", n_features=50_000),
#     TfidfTransformer(),
#     TruncatedSVD(n_components=100, random_state=0),
#     Normalizer(copy=False),
# )

# t0 = time()
# X_hashed_lsa = lsa_vectorizer.fit_transform(descriptions)
# print(f"vectorization done in {time() - t0:.3f} s")
# fit_and_evaluate(kmeans, X_hashed_lsa, name="KMeans\nwith LSA on hashed vectors")
# fit_and_evaluate(
#     minibatch_kmeans,
#     X_hashed_lsa,
#     name="MiniBatchKMeans\nwith LSA on hashed vectors",
# )
# import matplotlib.pyplot as plt
# import pandas as pd

# fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(16, 6), sharey=True)

# df = pd.DataFrame(evaluations[::-1]).set_index("estimator")
# df_std = pd.DataFrame(evaluations_std[::-1]).set_index("estimator")
# print("df", df)
# print("df_std", df_std)

# df.drop(
#     ["train_time"],
#     axis="columns",
# ).plot.barh(ax=ax0, xerr=df_std)
# ax0.set_xlabel("Clustering scores")
# ax0.set_ylabel("")

# df["train_time"].plot.barh(ax=ax1, xerr=df_std["train_time"])
# ax1.set_xlabel("Clustering time (s)")
# plt.tight_layout()
# plt.show()
