import numpy as np
import os
import math
from pprint import pprint
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import aggregate_classes
import metric_counts


def calculate_best_normalized_entropy(topic_counts, metric):
    best_score = -1  # Initialize with a value lower than the possible range (0 to 1)
    best_topic = None

    for topic_num, topic_data in topic_counts.items():
        category_counts = topic_data[metric]
        total_count = sum(category_counts.values())

        if total_count == 0:
            continue  # Skip topics with no counts to avoid division by zero

        entropy = 0
        for count in category_counts.values():
            if count > 0:
                p = count / total_count
                entropy -= p * math.log2(p)

        # Normalize the entropy
        max_entropy = math.log2(len(category_counts))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1
        score = 1 - normalized_entropy  # Higher score means lower entropy

        if score > best_score:
            best_score = score
            best_topic = topic_num

    if best_topic is None:
        return None, 0  # Return None and 0 if no valid topics were found

    return best_topic, best_score


def calculate_nmi_for_metric(topic_assignments, true_labels):
    """
    Calculate Normalized Mutual Information score.

    :param topic_assignments: List of topic assignments for each document
    :param true_labels: List of true category labels for each document
    :return: NMI score
    """
    return adjusted_mutual_info_score(true_labels, topic_assignments)


def calculate_v_measure(topic_assignments, true_labels):
    """
    Calculate V-measure and its components for topic modeling results.

    :param topic_assignments: List of topic assignments for each document
    :param true_labels: List of true category labels for each document
    :return: Dictionary containing homogeneity, completeness, and V-measure scores
    """
    homogeneity = homogeneity_score(true_labels, topic_assignments)
    completeness = completeness_score(true_labels, topic_assignments)
    v_measure = v_measure_score(true_labels, topic_assignments)

    return {
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure,
    }


def evaluate_topic_model_for_metric(topic_counts, balanced_counts, metric):
    """
    Evaluate the topic model for a single metric.

    :param topic_counts: Dictionary of topic assignments and counts for the metric
    :param balanced_counts: Dictionary of balanced counts for the metric's classes
    :param metric: The name of the metric being evaluated
    :return: NMI score for the metric
    """
    true_labels = []
    topic_assignments = []

    # Create true labels based on balanced counts
    for category, count in balanced_counts[metric].items():
        true_labels.extend([category] * count)

    # Create topic assignments based on topic counts
    for topic, counts in topic_counts.items():
        for category, count in counts[metric].items():
            topic_assignments.extend([topic] * count)

    # Ensure both lists have the same length
    min_length = min(len(true_labels), len(topic_assignments))
    true_labels = true_labels[:min_length]
    topic_assignments = topic_assignments[:min_length]

    return calculate_v_measure(topic_assignments, true_labels)


def calculate_purity(topic_counts, metric, score):
    # Find the best topic for this metric
    best_topic_num = -1
    best_count = 0
    best_topic_data = {}
    for topic_num, topic_count in topic_counts.items():
        if topic_count[metric][score] > best_count:
            best_count = topic_count[metric][score]
            best_topic_num = topic_num
            best_topic_data = topic_count

    total_count = sum(best_topic_data[metric].values())
    max_category_count = max(best_topic_data[metric].values())

    purity = max_category_count / total_count

    return best_topic_num, purity


def calculate_f1_score(topic_counts, ground_truth_counts, metric, score):

    # Find the best topic for this metric
    best_topic_num = -1
    best_count = 0
    best_topic_data = {}
    for topic_num, topic_count in topic_counts.items():
        if topic_count[metric][score] > best_count:
            best_count = topic_count[metric][score]
            best_topic_num = topic_num
            best_topic_data = topic_count

    true_labels = []
    predicted_labels = []

    for category, count in ground_truth_counts[metric].items():
        true_labels.extend([category] * count)
        predicted_category = max(
            best_topic_data[metric], key=best_topic_data[metric].get
        )
        predicted_labels.extend([predicted_category] * count)

    f1 = f1_score(true_labels, predicted_labels, average="weighted")

    return best_topic_num, f1


def evaluate_topic_model(topic_counts, ground_truth_data, metric, num_topics):
    ground_truth_counts = metric_counts.calculate_metric_counts(ground_truth_data, 3)
    results = {metric: {}}
    for score in ground_truth_counts[metric]:
        best_topic_purity, purity = calculate_purity(topic_counts, metric, score)
        best_topic_entropy, entropy = calculate_best_normalized_entropy(
            topic_counts, metric
        )
        v_measure = evaluate_topic_model_for_metric(
            topic_counts, ground_truth_counts, metric
        )
        results[metric][score] = {
            "best_topic_purity": best_topic_purity,
            "purity": purity,
            "best_topic_entropy": best_topic_entropy,
            "entropy": entropy,
        }

    print(f"Metric: {metric}, Topics: {num_topics} entropy score {v_measure}")

    return results


def plot_topic_model_performance(topic_numbers, evaluation_results, metric):
    import config

    prefix = ""
    if not config.balanced:
        prefix = "unbalanced/"
    metrics = list(evaluation_results[topic_numbers[0]].keys())

    # Separate plots for Purity and F1 Score
    for measure in ["purity", "entropy"]:
        fig, axs = plt.subplots(
            len(metrics), 1, figsize=(15, 6 * len(metrics)), squeeze=False
        )
        fig.suptitle(
            f"Topic Model Performance - {measure.capitalize()} Across Different Numbers of Topics",
            fontsize=20,
            fontweight="bold",
        )

        for i, metric in enumerate(metrics):
            ax = axs[i, 0]
            scores = evaluation_results[topic_numbers[0]][metric]
            categories = list(scores.keys())

            for category in categories:
                category_scores = [
                    evaluation_results[n][metric][category][measure]
                    for n in topic_numbers
                ]
                ax.plot(topic_numbers, category_scores, marker="o", label=category)

            ax.set_title(f"{metric} - {measure.capitalize()}")
            ax.set_xlabel("Number of Topics")
            ax.set_ylabel(f"{measure.capitalize()} Score")
            ax.legend(title="Categories", bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        outpath = f"./topic_eval/{prefix}topic_model_performance_{measure}_ground_truth_{metric}.png"
        print(f"outputting graph to: {outpath}")
        plt.savefig(
            outpath,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


topic_numbers = [14, 16, 18, 20, 50, 75, 100]
metric_names = [
    "privilegesRequired",
    "userInteraction",
    "confidentialityImpact",
    "integrityImpact",
    "availabilityImpact",
]

# To compare different numbers of topics:
for metric in metric_names:
    all_evaluation_results = {
        n: evaluate_topic_model(
            *aggregate_classes.create_topic_data(current_metric=metric, num_topics=n),
            metric,
            n,
        )
        for n in topic_numbers
    }

    plot_topic_model_performance(topic_numbers, all_evaluation_results, metric)
