import metric_counts
from pprint import pprint
import json


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


def main():
    x = range(2, 5, 1)

    topic_groups = read_data(
        f"./lda_word2vec_desc_compare_output/lda_word2vec_topics.json"
    )["topic_groups"]
    topics = extract_topics(topic_groups)

    for i in x:
        print(f"topic_assignments_{i}")
        print(
            "**************************************************************************"
        )
        print(
            "**************************************************************************"
        )
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
            print("Top 10 topic words")
            print(topics[i][topic])
            print()
            pprint(topic_counts[topic])
            print(
                "--------------------------------------------------------------------------"
            )

            topic_counts[topic]["topic_words"] = topics[i][topic]

        print(
            "**************************************************************************"
        )
        print(
            "**************************************************************************"
        )


if __name__ == "__main__":
    main()
