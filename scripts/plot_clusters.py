import matplotlib.pyplot as plt
import numpy as np


# Data
data = {
    0: {
        "attackComplexity": {"HIGH": 2756, "LOW": 34507},
        "attackVector": {
            "ADJACENT_NETWORK": 1400,
            "LOCAL": 12236,
            "NETWORK": 22757,
            "PHYSICAL": 870,
        },
        "availabilityImpact": {"HIGH": 22907, "LOW": 707, "NONE": 13649},
        "confidentialityImpact": {"HIGH": 26985, "LOW": 4233, "NONE": 6045},
        "integrityImpact": {"HIGH": 22283, "LOW": 3055, "NONE": 11925},
        "privilegesRequired": {"HIGH": 3514, "LOW": 13969, "NONE": 19780},
        "scope": {"CHANGED": 2858, "UNCHANGED": 34405},
        "userInteraction": {"NONE": 30471, "REQUIRED": 6792},
        "topic_words": [
            "user",
            "allow",
            "access",
            "remote",
            "code",
            "information",
            "arbitrary",
            "execute",
            "versions",
            "exploit",
        ],
    },
    1: {
        "attackComplexity": {"HIGH": 1289, "LOW": 24877},
        "attackVector": {
            "ADJACENT_NETWORK": 221,
            "LOCAL": 8421,
            "NETWORK": 17331,
            "PHYSICAL": 193,
        },
        "availabilityImpact": {"HIGH": 13580, "LOW": 823, "NONE": 11763},
        "confidentialityImpact": {"HIGH": 15976, "LOW": 4111, "NONE": 6079},
        "integrityImpact": {"HIGH": 12418, "LOW": 3329, "NONE": 10419},
        "privilegesRequired": {"HIGH": 2096, "LOW": 8571, "NONE": 15499},
        "scope": {"CHANGED": 2503, "UNCHANGED": 23663},
        "userInteraction": {"NONE": 18611, "REQUIRED": 7555},
        "topic_words": [
            "user",
            "allow",
            "access",
            "remote",
            "code",
            "information",
            "arbitrary",
            "execute",
            "versions",
            "exploit",
        ],
    },
}


def calculate_percentages(cluster_data):
    return {
        metric: {
            cat: value / sum(cluster_data[metric].values()) * 100
            for cat, value in cluster_data[metric].items()
        }
        for metric in cluster_data
        if metric != "topic_words"
    }


data[0] = {**calculate_percentages(data[0]), "topic_words": data[0]["topic_words"]}
data[1] = {**calculate_percentages(data[1]), "topic_words": data[1]["topic_words"]}


def plot_comparison(metric):
    categories = list(data[0][metric].keys())
    cluster0_values = [data[0][metric][cat] for cat in categories]
    cluster1_values = [data[1][metric][cat] for cat in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, cluster0_values, width, label="Cluster 0")
    rects2 = ax.bar(x + width / 2, cluster1_values, width, label="Cluster 1")

    ax.set_ylabel("Percentage")
    ax.set_title(f"Comparison of {metric}")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend()

    ax.bar_label(rects1, fmt="%.1f%%", padding=3)
    ax.bar_label(rects2, fmt="%.1f%%", padding=3)

    fig.tight_layout()
    plt.savefig(f"./temp_plots/{metric}_comparison_percentage.png")
    plt.close()


def plot_all_metrics_grid():
    metrics = [m for m in data[0].keys() if m != "topic_words"]
    n_metrics = len(metrics)

    fig, axs = plt.subplots(3, 3, figsize=(20, 20), sharex=False, sharey=False)
    fig.suptitle(
        "Comparison of All Metrics Between Clusters (in Percentages)", fontsize=16
    )

    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3

        categories = list(data[0][metric].keys())
        cluster0_values = [data[0][metric][cat] for cat in categories]
        cluster1_values = [data[1][metric][cat] for cat in categories]

        x = np.arange(len(categories))
        width = 0.35

        axs[row, col].bar(x - width / 2, cluster0_values, width, label="Cluster 0")
        axs[row, col].bar(x + width / 2, cluster1_values, width, label="Cluster 1")

        axs[row, col].set_ylabel("Percentage")
        axs[row, col].set_title(metric)
        axs[row, col].set_xticks(x)
        axs[row, col].set_xticklabels(categories, rotation=45, ha="right")
        axs[row, col].legend()

        # Add percentage labels on the bars
        for j, v in enumerate(cluster0_values):
            axs[row, col].text(
                x[j] - width / 2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8
            )
        for j, v in enumerate(cluster1_values):
            axs[row, col].text(
                x[j] + width / 2, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8
            )

        # Set y-axis to go from 0 to 100%
        axs[row, col].set_ylim(0, 100)

    # Add topic words to the empty subplot
    topic_words_text = (
        "Top Topic Words:\n\nCluster 0:\n"
        + ", ".join(data[0]["topic_words"])
        + "\n\nCluster 1:\n"
        + ", ".join(data[1]["topic_words"])
    )
    axs[2][2].text(0.5, 0.5, topic_words_text, ha="center", va="center", wrap=True)
    axs[2][2].axis("off")

    plt.tight_layout()
    plt.savefig(
        "./temp_plots/all_metrics_comparison_grid_with_topics_percentage.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# Generate individual plots for each metric
for metric in data[0].keys():
    if metric != "topic_words":
        plot_comparison(metric)

# Generate the combined plot with all metrics in a grid layout
plot_all_metrics_grid()

print("All individual comparison graphs have been generated and saved as PNG files.")
print(
    "The combined graph with all metrics in a grid layout, topic words, and percentages has been saved as 'all_metrics_comparison_grid_with_topics_percentage.png'."
)
