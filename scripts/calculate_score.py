import numpy as np
from sklearn.metrics import f1_score
from collections import Counter
import aggregate_classes

num_topics = [14, 16, 18, 20, 50, 75, 100]


def calculate_purity_balanced(topic_counts, balanced_counts, metric):
    categories = list(balanced_counts[metric].keys())

    # Find the best topic for this metric
    best_topic = max(topic_counts.items(), key=lambda x: max(x[1][metric].values()))

    best_topic_num, best_topic_data = best_topic

    total_count = sum(best_topic_data[metric].values())
    max_category_count = max(best_topic_data[metric].values())

    purity = max_category_count / total_count

    return best_topic_num, purity


def calculate_f1_score_balanced(topic_counts, balanced_data, metric):
    categories = list(balanced_data[metric].keys())

    # Find the best topic for this metric
    best_topic = max(topic_counts.items(), key=lambda x: max(x[1][metric].values()))

    best_topic_num, best_topic_data = best_topic

    # Create true labels and predicted labels
    true_labels = []
    predicted_labels = []

    for category, count in balanced_data[metric].items():
        true_labels.extend([category] * count)
        predicted_category = max(
            best_topic_data[metric], key=best_topic_data[metric].get
        )
        predicted_labels.extend([predicted_category] * count)

    # Calculate F1 score
    f1 = f1_score(true_labels, predicted_labels, average="weighted")

    return best_topic_num, f1


def evaluate_topic_model_balanced(topic_counts, balanced_data):
    metrics = [metric for metric in balanced_data.keys() if metric != "topic_words"]

    results = {}

    for metric in metrics:
        best_topic_purity, purity = calculate_purity_balanced(
            topic_counts, balanced_data, metric
        )
        best_topic_f1, f1 = calculate_f1_score_balanced(
            topic_counts, balanced_data, metric
        )

        results[metric] = {
            "best_topic_purity": best_topic_purity,
            "purity": purity,
            "best_topic_f1": best_topic_f1,
            "f1_score": f1,
        }

    return results


# Function to plot the results
def plot_topic_model_performance_balanced(topic_numbers, evaluation_results):
    metrics = list(evaluation_results[topic_numbers[0]].keys())

    fig, axs = plt.subplots(len(metrics), 2, figsize=(15, 5 * len(metrics)))
    fig.suptitle(
        "Topic Model Performance Across Different Numbers of Topics (Balanced Data)"
    )

    for i, metric in enumerate(metrics):
        purity_scores = [
            results[metric]["purity"] for results in evaluation_results.values()
        ]
        f1_scores = [
            results[metric]["f1_score"] for results in evaluation_results.values()
        ]

        axs[i, 0].plot(topic_numbers, purity_scores, marker="o")
        axs[i, 0].set_title(f"{metric} - Purity")
        axs[i, 0].set_xlabel("Number of Topics")
        axs[i, 0].set_ylabel("Purity Score")

        axs[i, 1].plot(topic_numbers, f1_scores, marker="o")
        axs[i, 1].set_title(f"{metric} - F1 Score")
        axs[i, 1].set_xlabel("Number of Topics")
        axs[i, 1].set_ylabel("F1 Score")

    plt.tight_layout()
    plt.savefig("topic_model_performance_balanced.png")
    plt.close()


# Example usage:
# balanced_data = {
#     "attackComplexity": {"HIGH": 20000, "LOW": 20000},
#     "attackVector": {"NETWORK": 20000, "ADJACENT_NETWORK": 20000, "LOCAL": 20000, "PHYSICAL": 20000},
#     # ... other metrics ...
# }
# evaluation_results = evaluate_topic_model_balanced(topic_counts, balanced_data)
# for metric, scores in evaluation_results.items():
#     print(f"{metric}:")
#     print(f"  Best Topic (Purity): {scores['best_topic_purity']}, Purity: {scores['purity']:.4f}")
#     print(f"  Best Topic (F1): {scores['best_topic_f1']}, F1 Score: {scores['f1_score']:.4f}")
#     print()

num_topics = [14, 16, 18, 20, 50, 75, 100]
# To compare different numbers of topics:
# topic_numbers = [10, 20, 30, 40, 50]
# all_evaluation_results = {n: evaluate_topic_model_balanced(...) for n in topic_numbers}
# plot_topic_model_performance_balanced(topic_numbers, all_evaluation_results)
