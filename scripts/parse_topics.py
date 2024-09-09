import re
from pprint import pprint
from typing import Dict, List, Set

TopicDict = Dict[int, List[str]]

pattern = r"Topic (\d+):"
sentinel = "THISISTHEEND"


def next_item(iterator):
    val = next(iterator)
    return val, val != sentinel


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
            current_topic = int(match.group(1))
            topics[current_topic] = []
        elif "-" in current:
            topics[current_topic].append(current.split("-")[1])


def parse_name(name: str):

    sections = name.split("_")
    seed = list(filter(lambda x: "seed" in x, sections))[0]
    seed = seed.split(")")[0]
    rv = seed, sections[2][1]
    return rv


def check_duplicates(run: TopicDict) -> Dict[str, int]:
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


def parse_topics():
    runs = {"seeds": {}, "topics": {}}
    path = "../data/results/new_results/lda_word2vec_desc_compare_output_seeds/lda_word2vec_output_seeds_all_metrics_all_values.txt"
    with open(path, "r") as f:
        contents = f.read()
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
                            runs["topics"][num_topics] = {}

                        run = parse_run(iterator)
                        runs["seeds"][seed][num_topics] = run
                        runs["topics"][num_topics][seed] = run
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


def main():
    runs = parse_topics()
    count_dupes(runs)


if __name__ == "__main__":
    main()
