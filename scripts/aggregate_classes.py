import metric_counts
import plot
import json
import utils


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

    topic_groups = read_data(
        f"../data/results/new_results/lda_word2vec_desc_compare_output/lda_word2vec_topics.json"
    )["topic_groups"]
    topics = extract_topics(topic_groups)

    for i in x:
        data = read_data(f"../results/lda_compare/topic_assignments_{i}.json")
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
                topic_data[topic], 3, "metric_value"
            )
            topic_counts[topic]["topic_words"] = topics[i][topic]
        version = 31
        nvd_data = utils.read_data(f"../data/nvd_{version}_cleaned.pkl")["data"]
        nvd_counts = metric_counts.calculate_metric_counts(nvd_data, version)
        plot.plot_all_metrics_gt_grid(topic_counts, nvd_counts)


if __name__ == "__main__":
    main()
