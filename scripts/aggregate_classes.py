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


def main():
    # plot_best_clusters()
    # x = range(2, 5, 1)
    x = range(1, 2, 1)

    # topic_groups = read_data(
    #     f"../data/results/new_results/lda_word2vec_desc_compare_output/lda_word2vec_topics.json"
    # )["topic_groups"]
    # topic_groups = read_data(
    #     f"../data/results/lda_word2vec_desc_compare_output_seeds/lda_seeds_topics.json"
    # )["topic_groups"]
    # topics = extract_topics(topic_groups)

    import num_topics

    version = 31
    num_topics = num_topics.num_topics
    current_metric = "confidentialityImpact"

    output_dir = f"./temp_plots/counts_CA_{num_topics}/"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(1, 6):
        os.makedirs(f"{output_dir}/{i}", exist_ok=True)

    topic_groups = read_data(
        f"./lda_word2vec_balanced_CA_{num_topics}/lda_balanced_topics.json"
    )["topic_groups"]
    # topic_groups = read_data(
    #     f"./lda_word2vec_balanced_fasttext_{num_topics}/lda_balanced_topics.json"
    # )["topic_groups"]
    # topic_groups = read_data(
    #     f"./kmeans_fasttext_{num_topics}/kmeans_balanced_topics.json"
    # )["topic_groups"]
    topics = extract_topics(topic_groups)
    for i in x:
        # data = read_data(f"../results/lda_compare/topic_assignments_{i}.json")
        # data = read_data(
        #     f"../data/results/lda_word2vec_desc_compare_output_seeds/topic_assignments_{i}.json"
        # )
        # data = read_data(
        #     f"./lda_word2vec_balanced_{num_topics}/topic_assignments_lda_model_t{num_topics}_asymmetric_e0.1_p30_i200_seed0.json"
        # )

        # data = read_data(
        #     f"./lda_word2vec_balanced_fasttext_{num_topics}/topic_assignments_lda_model_t{num_topics}_asymmetric_e0.1_p30_i200_seed0.json"
        # )
        data = read_data(
            f"./lda_word2vec_balanced_CA_{num_topics}/topic_assignments_lda_model_t{num_topics}_asymmetric_e0.1_p30_i200_seed0.json"
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
        # visualize_topics(
        #     topic_counts,
        #     ["confidentialityImpact", "integrityImpact", "availabilityImpact"],
        #     num_topics,
        # )
        # nvd_data = utils.read_data(f"../data/nvd_{version}_cleaned.pkl")["data"]
        # nvd_counts = metric_counts.calculate_metric_counts(nvd_data, version)

        # metrics = metric_counts.v3_metrics_counts()
        # for i in range(1, 6):
        #     for category in metrics[current_metric]:
        #         plot.plot_merged_top_k_topics_category_focus_counts(
        #             topic_counts,
        #             current_metric,
        #             category,
        #             k=i,
        #             num_topics=num_topics,
        #         )


if __name__ == "__main__":
    main()
