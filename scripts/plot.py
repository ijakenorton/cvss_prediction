import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict, Counter


def plot_comparison(nvd, mitre):

    # Creating subplots
    fig, axs = plt.subplots(4, 2, figsize=(14, 20))
    fig.tight_layout(pad=6.0)

    # Setting the overall title
    fig.suptitle("Summary of Security Metrics", fontsize=16, fontweight="bold")

    # Plotting each dictionary as a bar chart
    categories = list(nvd.keys())
    width = 0.35  # the width of the bars

    for ax, category in zip(axs.flat, categories):
        nvd_values = list(nvd[category].values())
        mitre_values = list(mitre[category].values())
        labels = list(nvd[category].keys())

        # Set positions of the bars
        x = range(len(labels))  # label location
        ax.bar(x, nvd_values, width, label="NVD", color="#377eb8")
        ax.bar(
            [p + width for p in x], mitre_values, width, label="MITRE", color="#ff7f00"
        )

        ax.set_title(category)
        ax.set_ylabel("Counts")
        ax.set_xlabel("Categories")
        ax.set_xticks([p + width / 2 for p in x])
        ax.set_xticklabels(labels)
        ax.legend()

    plt.show()


def plot_metrics_percentage(data):
    # Creating subplots
    fig, axs = plt.subplots(4, 2, figsize=(14, 20))
    fig.tight_layout(pad=6.0)
    # Setting the overall title
    fig.suptitle("Summary of Security Metrics", fontsize=16, fontweight="bold")

    # Plotting each dictionary as a bar chart
    for ax, (key, value) in zip(axs.flat, data.items()):
        nvd_values = list(data[key].values())
        nvd_total = sum(nvd_values)
        nvd_percentages = [(value / nvd_total) * 100 for value in nvd_values]
        ax.bar(value.keys(), nvd_percentages, color=plt.cm.Paired.colors)
        ax.set_title(key)
        ax.set_ylabel("Percentage of Category")
        ax.set_xlabel("Categories")

        # Set y-axis limits to 0-100
        ax.set_ylim(0, 100)

        # Add percentage labels on top of each bar
        for i, v in enumerate(nvd_percentages):
            ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom")

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the figure
    plt.savefig(
        "./temp_plots/nvd_metrics_percentage.png",
        bbox_inches="tight",  # This ensures that the entire figure is saved
    )

    # Close the figure to free up memory
    plt.close(fig)


def plot_percentages(nvd, mitre):
    # Creating subplots
    fig, axs = plt.subplots(4, 2, figsize=(14, 20))
    fig.tight_layout(pad=6.0)

    # Setting the overall title
    fig.suptitle(
        "Summary of Security Metrics (Percentages)", fontsize=16, fontweight="bold"
    )

    # Plotting each dictionary as a bar chart
    categories = list(nvd.keys())
    width = 0.35  # the width of the bars

    for ax, category in zip(axs.flat, categories):
        nvd_values = list(nvd[category].values())
        mitre_values = list(mitre[category].values())
        labels = list(nvd[category].keys())

        # Calculate percentages
        nvd_total = sum(nvd_values)
        mitre_total = sum(mitre_values)

        nvd_percentages = [(value / nvd_total) * 100 for value in nvd_values]
        mitre_percentages = [(value / mitre_total) * 100 for value in mitre_values]

        # Set positions of the bars
        x = range(len(labels))  # label location
        # Using color-blind friendly colors: blue and orange
        bars1 = ax.bar(x, nvd_percentages, width, label="NVD", color="#377eb8")  # Blue
        bars2 = ax.bar(
            [p + width for p in x],
            mitre_percentages,
            width,
            label="MITRE",
            color="#ff7f00",
        )  # Orange

        # Adding percentage labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}%",
                ha="center",
                va="bottom",
            )

        for bar in bars2:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}%",
                ha="center",
                va="bottom",
            )

        ax.set_title(category)
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel("Categories")
        ax.set_xticks([p + width / 2 for p in x])
        ax.set_xticklabels(labels)
        ax.legend()
    plt.show()


def plot_metrics(data):
    # Creating subplots
    fig, axs = plt.subplots(4, 2, figsize=(14, 20))
    fig.tight_layout(pad=6.0)

    # Setting the overall title
    fig.suptitle("Summary of Security Metrics", fontsize=16, fontweight="bold")

    # Plotting each dictionary as a bar chart
    for ax, (key, value) in zip(axs.flat, data.items()):
        ax.bar(value.keys(), value.values(), color=plt.cm.Paired.colors)
        ax.set_title(key)
        ax.set_ylabel("Counts")
        ax.set_xlabel("Categories")

    plt.show()


def calculate_percentages(cluster_data):
    return {
        metric: {
            cat: value / sum(cluster_data[metric].values()) * 100
            for cat, value in cluster_data[metric].items()
        }
        for metric in cluster_data
        if metric != "topic_words"
    }


def plot_all_metrics_gt_grid(
    topic_counts,
    nvd_counts,
    path="./temp_plots/all_metrics_comparison_grid_with_topics_percentage_gt",
):
    topic_counts = OrderedDict(sorted(topic_counts.items(), key=lambda x: int(x[0])))
    data = {
        topic: calculate_percentages(counts) for topic, counts in topic_counts.items()
    }
    num_clusters = len(data)
    metrics = [
        metric for metric in next(iter(data.values())).keys() if metric != "topic_words"
    ]
    n_metrics = len(metrics)

    rows = int(np.ceil(np.sqrt(n_metrics + 1)))  # +1 for topic words
    cols = int(np.ceil((n_metrics + 1) / rows))

    fig, axs = plt.subplots(
        rows, cols, figsize=(5 * cols, 5 * rows), sharex=False, sharey=False
    )
    fig.suptitle(
        f"Comparison of All Metrics Between {num_clusters} Clusters (in Percentages)",
        fontsize=16,
    )

    for i, metric in enumerate(metrics):
        row = i // cols
        col = i % cols

        categories = list(next(iter(data.values()))[metric].keys())
        values = [cluster_data[metric] for cluster_data in data.values()]
        nvd_values = [
            nvd_counts[metric][cat] / sum(nvd_counts[metric].values()) * 100
            for cat in categories
        ]

        x = np.arange(len(categories))
        width = 0.8 / (num_clusters + 1)  # +1 for NVD data

        for j, (topic, cluster_data) in enumerate(data.items()):
            cluster_values = [cluster_data[metric][cat] for cat in categories]
            axs[row, col].bar(
                x + j * width - num_clusters * width / 2,
                cluster_values,
                width,
                label=f"Cluster {topic}",
            )

        # Add NVD data bar
        axs[row, col].bar(
            x + num_clusters * width - num_clusters * width / 2,
            nvd_values,
            width,
            label="NVD Data",
            color="black",
        )

        axs[row, col].set_ylabel("Percentage")
        axs[row, col].set_title(metric)
        axs[row, col].set_xticks(x)
        axs[row, col].set_xticklabels(categories, rotation=90, ha="center")
        axs[row, col].legend(fontsize="x-small")

        for j, (topic, cluster_data) in enumerate(data.items()):
            cluster_values = [cluster_data[metric][cat] for cat in categories]
            for k, v in enumerate(cluster_values):
                axs[row, col].text(
                    x[k] + j * width - num_clusters * width / 2,
                    v,
                    f"{v:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )

        # Add NVD data labels
        for k, v in enumerate(nvd_values):
            axs[row, col].text(
                x[k] + num_clusters * width - num_clusters * width / 2,
                v,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
            )

        axs[row, col].set_ylim(0, 100)

    # Add topic words to the last subplot
    topic_words_text = "Top Topic Words:\n\n" + "\n\n".join(
        [
            f"Cluster {topic}:\n" + ", ".join(cluster_data["topic_words"])
            for topic, cluster_data in topic_counts.items()
        ]
    )
    axs[-1, -1].text(0.5, 0.5, topic_words_text, ha="center", va="center", wrap=True)
    axs[-1, -1].axis("off")

    # Remove any unused subplots
    for i in range(n_metrics + 1, rows * cols):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout()
    plt.savefig(
        f"{path}_{num_clusters}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_all_metrics_grid(topic_counts):
    topic_counts = OrderedDict(sorted(topic_counts.items(), key=lambda x: int(x[0])))
    data = {
        topic: calculate_percentages(counts) for topic, counts in topic_counts.items()
    }
    num_clusters = len(data)
    metrics = [
        metric for metric in next(iter(data.values())).keys() if metric != "topic_words"
    ]
    n_metrics = len(metrics)

    rows = int(np.ceil(np.sqrt(n_metrics + 1)))  # +1 for topic words
    cols = int(np.ceil((n_metrics + 1) / rows))

    fig, axs = plt.subplots(
        rows, cols, figsize=(5 * cols, 5 * rows), sharex=False, sharey=False
    )
    fig.suptitle(
        f"Comparison of All Metrics Between {num_clusters} Clusters (in Percentages)",
        fontsize=16,
    )

    for i, metric in enumerate(metrics):
        row = i // cols
        col = i % cols

        categories = list(next(iter(data.values()))[metric].keys())
        values = [cluster_data[metric] for cluster_data in data.values()]

        x = np.arange(len(categories))
        width = 0.8 / num_clusters

        for j, (topic, cluster_data) in enumerate(data.items()):
            cluster_values = [cluster_data[metric][cat] for cat in categories]
            axs[row, col].bar(
                x + j * width - (num_clusters - 1) * width / 2,
                cluster_values,
                width,
                label=f"Cluster {topic}",
            )

        axs[row, col].set_ylabel("Percentage")
        axs[row, col].set_title(metric)
        axs[row, col].set_xticks(x)
        axs[row, col].set_xticklabels(categories, rotation=90, ha="center")
        axs[row, col].legend(fontsize="x-small")

        for j, (topic, cluster_data) in enumerate(data.items()):
            cluster_values = [cluster_data[metric][cat] for cat in categories]
            for k, v in enumerate(cluster_values):
                axs[row, col].text(
                    x[k] + j * width - (num_clusters - 1) * width / 2,
                    v,
                    f"{v:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )

        axs[row, col].set_ylim(0, 100)

    # Add topic words to the last subplot
    topic_words_text = "Top Topic Words:\n\n" + "\n\n".join(
        [
            f"Cluster {topic}:\n" + ", ".join(cluster_data["topic_words"])
            for topic, cluster_data in topic_counts.items()
        ]
    )
    axs[-1, -1].text(0.5, 0.5, topic_words_text, ha="center", va="center", wrap=True)
    axs[-1, -1].axis("off")

    # Remove any unused subplots
    for i in range(n_metrics + 1, rows * cols):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout()
    plt.savefig(
        f"./temp_plots/all_metrics_comparison_grid_with_topics_percentage{num_clusters}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
