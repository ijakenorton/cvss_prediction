import metric_counts
import plot
import json
import utils
import os
from visualizations import visualize_topics
from pprint import pprint


def read_data(path):
    with open(path, "r") as f:
        file = f.read()
        data = json.loads(file)
    return data


def extract_topics(topic_data):
    clean = {}
    for group in topic_data:
        num_topics = group["num_topics"]
        clean[num_topics] = {}
        topics = group["topics"]
        for topic in topics:
            clean[num_topics][topic["topic_id"]] = topic["words"]

    return clean


def percentage_plot():

    version = 31
    nvd_data = utils.read_data(f"../data/nvd_{version}_cleaned.pkl")["data"]
    nvd_counts = metric_counts.calculate_metric_counts(nvd_data, 31)
    plot.plot_metrics_percentage(nvd_counts)


def plot_best_clusters():
    topic_groups = read_data(f"./lda_word2vec_balanced/lda_18_topics.json")[
        "topic_groups"
    ]
    topics = extract_topics(topic_groups)
    data = read_data(
        f"./lda_word2vec_attack_vector/topic_assignments_lda_model_t18_asymmetric_e0.1_p30_i200_seed0.json"
    )

    best_topic_clusters = metric_counts.v3_metrics_counts()

    for metric in best_topic_clusters:
        for category in best_topic_clusters[metric]:
            best_topic_clusters[metric][category] = {
                "topic": -1,
                "count": 0,
                "topic_words": [],
            }

    topic_data = {}
    for desc in data:
        desc_topic = desc["topic"]
        if desc_topic in topic_data.keys():
            topic_data[desc_topic].append(desc)
        else:
            topic_data[desc_topic] = []

            topic_data[desc_topic].append(desc)
    topic_counts = {}
    for topic in topic_data:

        topic_counts[topic] = metric_counts.calculate_metric_counts(
            topic_data[topic], 3
        )

        # topic_counts[topic]["topic_words"] = topics[18][topic]

    for i in range(len(topic_counts)):
        for metric in topic_counts[i]:
            for category, count in topic_counts[i][metric].items():
                if count > best_topic_clusters[metric][category]["count"]:
                    best_topic_clusters[metric][category] = {
                        "topic": i,
                        "count": count,
                        # "topic_words": topic_counts[i]["topic_words"],
                    }

    for metric in best_topic_clusters:
        for category in best_topic_clusters[metric]:
            best_topic_clusters[metric][category]["topic_words"] = topics[18][
                best_topic_clusters[metric][category]["topic"]
            ]

    # topic_counts[topic] = metric_counts.calculate_metric_counts(
    #     topic_data[topic], 3
    # )

    version = 31
    pprint(best_topic_clusters)
    # nvd_data = utils.read_data(f"../data/nvd_{version}_cleaned.pkl")["data"]
    # nvd_counts = metric_counts.calculate_metric_counts(nvd_data, version)
    # plot.plot_all_metrics_gt_grid(topic_counts, nvd_counts, "18_only_best_topics")


def create_topic_data(current_metric=None, num_topics=None):
    import config

    if not current_metric:
        current_metric = config.current_metric

    if not num_topics:
        num_topics = config.num_topics
    prefix = ""
    balanced = "balanced"

    if not config.balanced:
        prefix = "unbalanced/"
        balanced = "unbalanced"

    output_dir = f"./temp_plots/{prefix}counts_{current_metric}_{num_topics}/"
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(1, 6):
        os.makedirs(f"{output_dir}/{i}", exist_ok=True)

    topic_groups = read_data(
        f"./{prefix}lda_word2vec_{balanced}_{current_metric}_{num_topics}/lda_{balanced}_topics.json"
    )["topic_groups"]

    topics = extract_topics(topic_groups)

    data = read_data(
        f"./{prefix}lda_word2vec_{balanced}_{current_metric}_{num_topics}/topic_assignments_lda_model_t{num_topics}_asymmetric_e0.1_p30_i200_seed0.json"
    )
    topic_data = {}
    for desc in data:
        if "topic" in desc.keys():
            desc_topic = desc["topic"]
        else:
            desc_topic = desc["cluster"]
        if desc_topic in topic_data.keys():
            topic_data[desc_topic].append(desc)
        else:
            topic_data[desc_topic] = []

            topic_data[desc_topic].append(desc)
    topic_counts = {}
    for topic in topic_data:

        topic_counts[topic] = metric_counts.calculate_metric_counts(
            topic_data[topic], 3
        )

        topic_counts[topic]["topic_words"] = topics[num_topics][topic]

    return topic_counts, data


def main():

    import config

    topic_counts, data = create_topic_data()

    # try:
    # visualize_topics(
    #     topic_counts,
    #     ["confidentialityImpact", "integrityImpact", "availabilityImpact"],
    #     config.num_topics,
    # )

    # nvd_data = utils.read_data(f"../data/nvd_{version}_cleaned.pkl")["data"]
    # nvd_counts = metric_counts.calculate_metric_counts(nvd_data, version)

    metrics = metric_counts.v3_metrics_counts()
    for i in range(1, 6):
        for category in metrics[config.current_metric]:
            plot.plot_merged_top_k_topics_category_focus_counts(
                topic_counts,
                config.current_metric,
                category,
                k=i,
                num_topics=config.num_topics,
            )


if __name__ == "__main__":
    main()
