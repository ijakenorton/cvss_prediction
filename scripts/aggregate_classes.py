import metric_counts
import plot
import json
import utils
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


def main():
    x = range(2, 5, 1)
    x = range(1, 2, 1)

    # topic_groups = read_data(
    #     f"../data/results/new_results/lda_word2vec_desc_compare_output/lda_word2vec_topics.json"
    # )["topic_groups"]
    # topic_groups = read_data(
    #     f"../data/results/lda_word2vec_desc_compare_output_seeds/lda_seeds_topics.json"
    # )["topic_groups"]
    # topics = extract_topics(topic_groups)

    topic_groups = read_data(f"./lda_word2vec_attack_vector/lda_18_topics.json")[
        "topic_groups"
    ]
    topics = extract_topics(topic_groups)
    for i in x:
        # data = read_data(f"../results/lda_compare/topic_assignments_{i}.json")
        # data = read_data(
        #     f"../data/results/lda_word2vec_desc_compare_output_seeds/topic_assignments_{i}.json"
        # )
        data = read_data(
            f"./lda_word2vec_attack_vector/topic_assignments_lda_model_t18_asymmetric_e0.1_p30_i200_seed0.json"
        )

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

            topic_counts[topic]["topic_words"] = topics[18][topic]

        attack_vector_topics = {
            "NETWORK": {"topic": -1, "count": 0, "topic_words": []},
            "ADJACENT_NETWORK": {"topic": -1, "count": 0, "topic_words": []},
            "LOCAL": {"topic": -1, "count": 0, "topic_words": []},
            "PHYSICAL": {"topic": -1, "count": 0, "topic_words": []},
        }
        for i in range(len(topic_counts)):
            for topic, count in topic_counts[i]["attackVector"].items():
                if count > attack_vector_topics[topic]["count"]:
                    attack_vector_topics[topic] = {
                        "topic": i,
                        "count": count,
                        "topic_words": topic_counts[i]["topic_words"],
                    }

        pprint(attack_vector_topics)
        version = 31
        nvd_data = utils.read_data(f"../data/nvd_{version}_cleaned.pkl")["data"]
        nvd_counts = metric_counts.calculate_metric_counts(nvd_data, version)
        plot.plot_all_metrics_gt_grid(topic_counts, nvd_counts, "explore_18")


if __name__ == "__main__":
    main()
