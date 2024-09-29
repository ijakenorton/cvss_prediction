import re
import json
from pprint import pprint
from typing import Dict, List, Set, Tuple

TopicDict = Dict[int, List[str]]

pattern = r"(Topic|Cluster) (\d+):"
sentinel = "THISISTHEEND"


def next_item(iterator):
    val = next(iterator)
    return val, val != sentinel


def parse_run_cluster(iterator):
    flag = True
    current_topic = -1
    topics = {}
    while flag:
        current, flag = next_item(iterator)
        if "Cluster assignments saved to:" in current:
            return topics
        match = re.search(pattern, current)
        if match:
            current_topic = int(match.group(2))
            topics[current_topic] = []
        elif "-" in current:
            word = current.split("-")[1].split(" ")[1]
            topics[current_topic].append(word)


def parse_run(iterator):
    flag = True
    current_topic = -1
    topics = {}
    while flag:
        current, flag = next_item(iterator)
        if "Topic assignments saved to:" in current:
            return topics
        match = re.search(pattern, current)
        if match:
            current_topic = int(match.group(2))
            topics[current_topic] = []
        elif "-" in current:
            topics[current_topic].append(current.split("-")[1])


def parse_name(name: str):

    sections = name.split("_")
    seed = list(filter(lambda x: "seed" in x, sections))[0]
    seed = seed.split(")")[0]
    rv = seed, sections[2][1]
    return rv


def check_duplicates(run: TopicDict) -> Tuple[Dict[str, int], int]:
    """
    Process a TopicDict.

    Example input:
        example: TopicDict = {
            0: [' versions', ' security'],
            1: [' code', ' vulnerabilities', ' flaw']
        }

    Example when extracting from topic run data:
        for run in runs["seeds"].values():
            for r in run.values():
                check_duplicates(r)

    Args:
        topic_dict (TopicDict): A dictionary mapping topic numbers to lists of related terms.

    Returns:
        None
    """

    comparison = {}
    total_dupes = 0
    for topics in run.values():
        for topic in topics:
            if topic in comparison.keys():
                comparison[topic] += 1
                total_dupes += 1
            else:
                comparison[topic] = 0

    return comparison, total_dupes


def parse_topics_kmeans(num_topics):

    path = f"./kmeans_fasttext_{num_topics}/kmeans_fasttext_all_metrics_all_values.txt"
    runs = {"seeds": {}, "num_topics": {}}
    with open(path, "r") as f:
        contents = f.read()
        lines = contents.split("\n")
        lines.append(sentinel)
        iterator = iter(lines)

        flag = True
        while flag:
            current, flag = next_item(iterator)
            if "Top 50 words for each cluster" in current:
                split_line = current.split(" ")
                for word in split_line:
                    if "kmeans" in word:
                        name = word
                        seed, num_topics = parse_name(name)
                        if seed not in runs.keys():
                            runs["seeds"][seed] = {}

                        if num_topics not in runs.keys():
                            runs["num_topics"][num_topics] = {}

                        run = parse_run_cluster(iterator)
                        runs["seeds"][seed][num_topics] = run
                        runs["num_topics"][num_topics][seed] = run
    return runs


def parse_topics(num_topics):

    # path = f"./lda_word2vec_balanced_{num_topics}/lda_word2vec_balanced_all_metrics_all_values.txt"
    # path = f"./lda_word2vec_balanced_fasttext_{num_topics}/lda_word2vec_balanced_all_metrics_all_values.txt"
    # path = f"./kmeans_fasttext_{num_topics}/kmeans_fasttext_all_metrics_all_values.txt"
    path = f"./lda_word2vec_balanced_CA_{num_topics}/lda_word2vec_balanced_all_metrics_all_values.txt"
    runs = {"seeds": {}, "num_topics": {}}
    with open(path, "r") as f:
        contents = f.read()
        print(contents)
        lines = contents.split("\n")
        lines.append(sentinel)
        iterator = iter(lines)

        flag = True
        while flag:
            current, flag = next_item(iterator)
            if "Top 50 words for each topic" in current:
                split_line = current.split(" ")
                for word in split_line:
                    if "lda" in word:
                        name = word
                        seed, num_topics = parse_name(name)
                        if seed not in runs.keys():
                            runs["seeds"][seed] = {}

                        if num_topics not in runs.keys():
                            runs["num_topics"][num_topics] = {}

                        run = parse_run(iterator)
                        runs["seeds"][seed][num_topics] = run
                        runs["num_topics"][num_topics][seed] = run
    return runs


def count_dupes(runs):
    duplicates = []
    totals = []
    for run in runs["seeds"].values():
        for r in run.values():
            dupe, total = check_duplicates(r)
            duplicates.append(dupe)
            totals.append(total)

    return duplicates, totals


def create_compatible_json(runs):
    seed0 = runs["seeds"]["seed0"]
    seed50 = runs["seeds"]["seed50"]
    seed100 = runs["seeds"]["seed100"]

    output = {"topic_groups": []}

    output["topic_groups"].append(
        {
            "num_topics": 2,
            "topics": [
                {"topic_id": 0, "words": seed50["2"][0]},
                {"topic_id": 1, "words": seed50["2"][1]},
            ],
        }
    )
    output["topic_groups"].append(
        {
            "num_topics": 3,
            "topics": [
                {"topic_id": 0, "words": seed0["3"][0]},
                {"topic_id": 1, "words": seed0["3"][1]},
                {"topic_id": 2, "words": seed0["3"][2]},
            ],
        }
    )
    output["topic_groups"].append(
        {
            "num_topics": 4,
            "topics": [
                {"topic_id": 0, "words": seed100["4"][0]},
                {"topic_id": 1, "words": seed100["4"][1]},
                {"topic_id": 2, "words": seed100["4"][2]},
                {"topic_id": 3, "words": seed100["4"][3]},
            ],
        }
    )
    with open(
        "../data/results/lda_word2vec_desc_compare_output_seeds/lda_seeds_topics.json",
        "w",
    ) as f:
        json.dump(output, f)


def create_single_topic(run, num_topics):
    path = f"./lda_word2vec_balanced_CA_{num_topics}/lda_balanced_topics.json"
    # path = f"./lda_word2vec_balanced_fasttext_{num_topics}/lda_balanced_topics.json"
    # path = f"./kmeans_fasttext_{num_topics}/kmeans_balanced_topics.json"

    output = {"topic_groups": []}

    topics = []
    for i, topic in enumerate(run.values()):

        topics.append({"topic_id": i, "words": topic})

    output["topic_groups"].append({"num_topics": len(topics), "topics": topics})
    with open(
        path,
        "w",
    ) as f:
        json.dump(output, f)


def main():

    import num_topics

    num_topics = num_topics.num_topics
    run_name = str(num_topics)[0]
    runs = parse_topics(num_topics)
    print(runs)
    # runs = parse_topics_kmeans(num_topics)
    run = runs["seeds"]["seed0"][run_name]
    create_single_topic(run, num_topics)
    # pprint(runs["seeds"]["seed50"])
    # create_compatible_json(runs)
    # dupes, count = count_dupes(runs)


if __name__ == "__main__":
    main()
