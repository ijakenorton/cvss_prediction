import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict, Counter, defaultdict


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

        # Addinlot_merged_top_k_category_focu percentage labels
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


def plot_merged_top_k_topics_category_focus_counts(
    topic_counts,
    metric,
    category,
    k=3,
    path="./temp_plots/counts/merged_top_k_topics_category_focus_counts",
    num_topics=18,
):
    path = f"./temp_plots/counts_{metric}_{num_topics}/{k}/merged_top_k_topics_category_focus_counts"
    categories = list(next(iter(topic_counts.values()))[metric].keys())

    # Find top k topics for the specified metric and category
    top_k_topics = sorted(
        topic_counts.items(),
        key=lambda x: x[1][metric][category],
        reverse=True,
    )[:k]

    merged_counts = []

    for cat in categories:
        # Merge data from top k topics
        merged_value = sum(topic_data[metric][cat] for _, topic_data in top_k_topics)
        merged_counts.append(merged_value)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(categories))
    width = 0.6  # Increased width since we only have one bar per category now

    rects = ax.bar(x, merged_counts, width, label=f"Top {k} Topics", color="skyblue")

    # Highlight the focus category
    focus_index = categories.index(category)
    ax.bar(x[focus_index], merged_counts[focus_index], width, color="blue")

    ax.set_ylabel("Count")
    ax.set_title(f"{metric} - Merged Top {k} Topics Counts (Focus on {category})")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend()

    # Add value labels on the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    autolabel(rects)

    # Add topic numbers and top words
    topic_info = f"Top {k} Topics for {category}: " + ", ".join(
        [f"{topic}" for topic, _ in top_k_topics]
    )
    topic_words = set()
    top_k = 5
    for _, topic_data in top_k_topics:
        if k == 1:
            top_k = 10
        topic_words.update(
            topic_data["topic_words"][:top_k]
        )  # Top 5 words from each topic
    topic_words_text = f"Common Topic Words:\n" + ", ".join(
        list(topic_words)[:15]
    )  # Limit to 15 words

    plt.text(
        0.5,
        -0.2,
        topic_info + "\n" + topic_words_text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        wrap=True,
        fontsize=8,
    )

    plt.tight_layout()
    plt.savefig(f"{path}_{metric}_{category}_k{k}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved as {path}_{metric}_{category}_k{k}.png")


# Example usage:
# plot_merged_top_k_topics_category_focus_counts(topic_counts, "attackComplexity", "HIGH", k=3)
def plot_merged_top_k_topics_category_focus_balanced(
    topic_counts,
    nvd_counts,
    metric,
    category,
    k=1,
    balanced_count=20000,
    path="./temp_plots/merged_top_k_topics_category_focus_balanced",
):
    categories = list(next(iter(topic_counts.values()))[metric].keys())

    # Find top k topics for the specified metric and category
    top_k_topics = sorted(
        topic_counts.items(),
        key=lambda x: x[1][metric][category] / sum(x[1][metric].values()),
        reverse=True,
    )[:k]

    merged_percentages = []
    nvd_percentages = []

    for cat in categories:
        # Merge data from top k topics
        merged_value = sum(topic_data[metric][cat] for _, topic_data in top_k_topics)
        merged_percentage = merged_value / (balanced_count * len(categories)) * 100
        merged_percentages.append(merged_percentage)

        # Use balanced count for NVD data
        nvd_percentage = balanced_count / (balanced_count * len(categories)) * 100
        nvd_percentages.append(nvd_percentage)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax.bar(
        x - width / 2,
        merged_percentages,
        width,
        label=f"Top {k} Topics",
        color="skyblue",
    )
    rects2 = ax.bar(
        x + width / 2, nvd_percentages, width, label="Balanced NVD Data", color="orange"
    )

    # Highlight the focus category
    focus_index = categories.index(category)
    ax.bar(
        x[focus_index] - width / 2, merged_percentages[focus_index], width, color="blue"
    )
    ax.bar(x[focus_index] + width / 2, nvd_percentages[focus_index], width, color="red")

    ax.set_ylabel("Percentage")
    ax.set_title(
        f"{metric} - Merged Top {k} Topics vs Balanced NVD Data (Focus on {category})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend()

    # Add value labels on the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    autolabel(rects1)
    autolabel(rects2)

    # Add topic numbers and top words
    topic_info = f"Top {k} Topics for {category}: " + ", ".join(
        [f"{topic}" for topic, _ in top_k_topics]
    )
    topic_words = set()
    for _, topic_data in top_k_topics:
        topic_words.update(topic_data["topic_words"][:5])  # Top 5 words from each topic
    topic_words_text = f"Common Topic Words:\n" + ", ".join(
        list(topic_words)[:15]
    )  # Limit to 15 words

    plt.text(
        0.5,
        -0.2,
        topic_info + "\n" + topic_words_text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        wrap=True,
        fontsize=8,
    )

    plt.tight_layout()
    plt.savefig(f"{path}_{metric}_{category}_k{k}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved as {path}_{metric}_{category}_k{k}.png")


# Example usage:
# plot_merged_top_k_topics_category_focus_balanced(topic_counts, nvd_counts, "attackComplexity", "HIGH", k=3, balanced_count=20000)


def plot_merged_top_k_topics_category_focus(
    topic_counts,
    nvd_counts,
    metric,
    category,
    k=3,
    path="./temp_plots/merged_top_k_topics_category_focus",
):
    categories = list(next(iter(topic_counts.values()))[metric].keys())

    # Find top k topics for the specified metric and category
    top_k_topics = sorted(
        topic_counts.items(),
        key=lambda x: x[1][metric][category] / sum(x[1][metric].values()),
        reverse=True,
    )[:k]

    merged_percentages = []
    nvd_percentages = []

    for cat in categories:
        # Merge data from top k topics
        merged_value = sum(topic_data[metric][cat] for _, topic_data in top_k_topics)
        merged_percentage = merged_value / sum(nvd_counts[metric].values()) * 100
        merged_percentages.append(merged_percentage)

        nvd_percentage = (
            nvd_counts[metric][cat] / sum(nvd_counts[metric].values()) * 100
        )
        nvd_percentages.append(nvd_percentage)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax.bar(
        x - width / 2,
        merged_percentages,
        width,
        label=f"Top {k} Topics",
        color="skyblue",
    )
    rects2 = ax.bar(
        x + width / 2, nvd_percentages, width, label="NVD Data", color="orange"
    )

    # Highlight the focus category
    focus_index = categories.index(category)
    ax.bar(
        x[focus_index] - width / 2, merged_percentages[focus_index], width, color="blue"
    )
    ax.bar(x[focus_index] + width / 2, nvd_percentages[focus_index], width, color="red")

    ax.set_ylabel("Percentage")
    ax.set_title(f"{metric} - Merged Top {k} Topics vs NVD Data (Focus on {category})")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend()

    # Add value labels on the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    autolabel(rects1)
    autolabel(rects2)

    # Add topic numbers and top words
    topic_info = f"Top {k} Topics for {category}: " + ", ".join(
        [f"{topic}" for topic, _ in top_k_topics]
    )
    topic_words = set()
    for _, topic_data in top_k_topics:
        topic_words.update(topic_data["topic_words"][:5])  # Top 5 words from each topic
    topic_words_text = f"Common Topic Words:\n" + ", ".join(
        list(topic_words)[:15]
    )  # Limit to 15 words

    plt.text(
        0.5,
        -0.2,
        topic_info + "\n" + topic_words_text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        wrap=True,
        fontsize=8,
    )

    plt.tight_layout()
    plt.savefig(f"{path}_{metric}_{category}_k{k}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved as {path}_{metric}_{category}_k{k}.png")


def plot_merged_top_k_topics_individual_with_total(
    topic_counts, nvd_counts, k=3, path="./temp_plots/merged_top_k_topics_max"
):
    metrics = [
        metric
        for metric in next(iter(topic_counts.values())).keys()
        if metric != "topic_words"
    ]

    for metric in metrics:
        categories = list(next(iter(topic_counts.values()))[metric].keys())

        for category in categories:
            # Find top k topics for this metric and category
            top_k_topics = sorted(
                topic_counts.items(),
                key=lambda x: x[1][metric][category] / sum(x[1][metric].values()),
                reverse=True,
            )[:k]

            # Merge data from top k topics
            merged_value = sum(
                topic_data[metric][category] for _, topic_data in top_k_topics
            )
            merged_total = sum(
                sum(topic_data[metric].values()) for _, topic_data in top_k_topics
            )
            merged_percentage = merged_value / merged_total * 100

            nvd_percentage = (
                nvd_counts[metric][category] / sum(nvd_counts[metric].values()) * 100
            )

            # Calculate total percentage against whole NVD dataset
            total_nvd_count = sum(sum(nvd_counts[m].values()) for m in metrics)
            merged_total_percentage = merged_value / total_nvd_count * 100
            nvd_total_percentage = nvd_counts[metric][category] / total_nvd_count * 100

            # Create the plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            fig.suptitle(
                f"{metric} - {category}\nMerged Top {k} Topics vs NVD Data", fontsize=16
            )

            x = np.arange(2)
            width = 0.35

            # First subplot: Percentage within the metric
            rects1 = ax1.bar(
                x[0], merged_percentage, width, label=f"Top {k} Topics", color="skyblue"
            )
            rects2 = ax1.bar(
                x[1], nvd_percentage, width, label="NVD Data", color="orange"
            )

            ax1.set_ylabel("Percentage within Metric")
            ax1.set_title("Comparison within Metric")
            ax1.set_xticks(x)
            ax1.set_xticklabels([f"Top {k} Topics", "NVD Data"])
            ax1.legend()

            # Second subplot: Percentage against whole dataset
            rects3 = ax2.bar(
                x[0],
                merged_total_percentage,
                width,
                label=f"Top {k} Topics",
                color="skyblue",
            )
            rects4 = ax2.bar(
                x[1], nvd_total_percentage, width, label="NVD Data", color="orange"
            )

            ax2.set_ylabel("Percentage of Total NVD Dataset")
            ax2.set_title("Comparison against Total NVD Dataset")
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"Top {k} Topics", "NVD Data"])
            ax2.legend()

            # Add value labels on the bars
            def autolabel(ax, rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(
                        f"{height:.2f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                    )

            autolabel(ax1, rects1)
            autolabel(ax1, rects2)
            autolabel(ax2, rects3)
            autolabel(ax2, rects4)

            # Add topic numbers and top words
            topic_info = f"Top {k} Topics: " + ", ".join(
                [f"{topic}" for topic, _ in top_k_topics]
            )
            topic_words = set()
            for _, topic_data in top_k_topics:
                topic_words.update(
                    topic_data["topic_words"][:5]
                )  # Top 5 words from each topic
            topic_words_text = f"Common Topic Words:\n" + ", ".join(
                list(topic_words)[:15]
            )  # Limit to 15 words
            plt.figtext(
                0.5,
                0.01,
                topic_info + "\n" + topic_words_text,
                ha="center",
                va="center",
                wrap=True,
            )

            plt.tight_layout()
            plt.savefig(
                f"{path}_{metric}_{category}_k{k}.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

            print(f"Plot saved as {path}_{metric}_{category}_k{k}.png")


def plot_merged_top_k_topics_all_categories(
    topic_counts,
    nvd_counts,
    k=3,
    path="./temp_plots/merged_top_k_topics_all_categories",
):
    metrics = [
        metric
        for metric in next(iter(topic_counts.values())).keys()
        if metric != "topic_words"
    ]

    for metric in metrics:
        categories = list(next(iter(topic_counts.values()))[metric].keys())

        merged_percentages = []
        nvd_percentages = []
        top_topics_info = {}

        for category in categories:
            # Find top k topics for this metric and category
            top_k_topics = sorted(
                topic_counts.items(),
                key=lambda x: x[1][metric][category] / sum(x[1][metric].values()),
                reverse=True,
            )[:k]

            # Merge data from top k topics
            merged_value = sum(
                topic_data[metric][category] for _, topic_data in top_k_topics
            )
            merged_percentage = merged_value / sum(nvd_counts[metric].values()) * 100
            merged_percentages.append(merged_percentage)

            nvd_percentage = (
                nvd_counts[metric][category] / sum(nvd_counts[metric].values()) * 100
            )
            nvd_percentages.append(nvd_percentage)

            # Store top topics info
            top_topics_info[category] = {
                "topics": [topic for topic, _ in top_k_topics],
                "words": set().union(
                    *[
                        set(topic_data["topic_words"][:5])
                        for _, topic_data in top_k_topics
                    ]
                ),
            }

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6 + 0.5 * len(categories)))

        x = np.arange(len(categories))
        width = 0.35

        rects1 = ax.bar(
            x - width / 2,
            merged_percentages,
            width,
            label=f"Top {k} Topics",
            color="skyblue",
        )
        rects2 = ax.bar(
            x + width / 2, nvd_percentages, width, label="NVD Data", color="orange"
        )

        ax.set_ylabel("Percentage")
        ax.set_title(f"{metric}\nMerged Top {k} Topics vs NVD Data")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.legend()

        # Add value labels on the bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        autolabel(rects1)
        autolabel(rects2)

        # Add topic numbers and top words
        topic_info_text = "\n".join(
            [
                f"{cat}: Topics {', '.join(map(str, info['topics']))}"
                for cat, info in top_topics_info.items()
            ]
        )
        all_words = set().union(*[info["words"] for info in top_topics_info.values()])
        topic_words_text = f"Common Topic Words: {', '.join(list(all_words)[:20])}"  # Limit to 20 words

        plt.text(
            0.5,
            -0.2 - 0.03 * len(categories),
            topic_info_text + "\n\n" + topic_words_text,
            ha="center",
            va="center",
            transform=ax.transAxes,
            wrap=True,
            fontsize=8,
        )

        plt.tight_layout()
        plt.savefig(f"{path}_{metric}_k{k}.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Plot saved as {path}_{metric}_k{k}.png")


def plot_merged_top_k_topics_individual(
    topic_counts, nvd_counts, k=3, path="./temp_plots/merged_top_k_topics_relative"
):
    metrics = [
        metric
        for metric in next(iter(topic_counts.values())).keys()
        if metric != "topic_words"
    ]

    for metric in metrics:
        categories = list(next(iter(topic_counts.values()))[metric].keys())

        for category in categories:
            # Find top k topics for this metric and category
            top_k_topics = sorted(
                topic_counts.items(),
                key=lambda x: x[1][metric][category] / sum(x[1][metric].values()),
                reverse=True,
            )[:k]

            # Merge data from top k topics
            merged_value = sum(
                topic_data[metric][category] for _, topic_data in top_k_topics
            )
            merged_total = sum(
                sum(topic_data[metric].values()) for _, topic_data in top_k_topics
            )
            # merged_percentage = merged_value / merged_total * 100
            merged_percentage = merged_value / sum(nvd_counts[metric].values()) * 100

            nvd_percentage = (
                nvd_counts[metric][category] / sum(nvd_counts[metric].values()) * 100
            )

            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))

            x = np.arange(2)
            width = 0.35

            rects1 = ax.bar(
                x[0], merged_percentage, width, label=f"Top {k} Topics", color="skyblue"
            )
            rects2 = ax.bar(
                x[1], nvd_percentage, width, label="NVD Data", color="orange"
            )

            ax.set_ylabel("Percentage")
            ax.set_title(f"{metric} - {category}\nMerged Top {k} Topics vs NVD Data")
            ax.set_xticks(x)
            ax.set_xticklabels([f"Top {k} Topics", "NVD Data"])
            ax.legend()

            # Add value labels on the bars
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(
                        f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                    )

            autolabel(rects1)
            autolabel(rects2)

            # Add topic numbers and top words
            topic_info = f"Top {k} Topics: " + ", ".join(
                [f"{topic}" for topic, _ in top_k_topics]
            )
            topic_words = set()
            for _, topic_data in top_k_topics:
                topic_words.update(
                    topic_data["topic_words"][:5]
                )  # Top 5 words from each topic
            topic_words_text = f"Common Topic Words:\n" + ", ".join(
                list(topic_words)[:15]
            )  # Limit to 15 words
            plt.text(
                0.5,
                -0.2,
                topic_info + "\n" + topic_words_text,
                ha="center",
                va="center",
                transform=ax.transAxes,
                wrap=True,
            )

            plt.tight_layout()
            plt.savefig(
                f"{path}_{metric}_{category}_k{k}.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

            print(f"Plot saved as {path}_{metric}_{category}_k{k}.png")


def plot_merged_top_k_topics(
    topic_counts, nvd_counts, k=3, path="./temp_plots/merged_top_k_topics"
):
    metrics = [
        metric
        for metric in next(iter(topic_counts.values())).keys()
        if metric != "topic_words"
    ]

    fig, axs = plt.subplots(
        len(metrics), 1, figsize=(12, 6 * len(metrics)), squeeze=False
    )
    fig.suptitle(f"Merged Top {k} Topics for Each Metric", fontsize=16)

    for idx, metric in enumerate(metrics):
        # Find top k topics for this metric
        top_k_topics = sorted(
            topic_counts.items(),
            key=lambda x: max(x[1][metric].values()) / sum(x[1][metric].values()),
            reverse=True,
        )[:k]

        # Merge data from top k topics
        merged_data = defaultdict(int)
        for _, topic_data in top_k_topics:
            for cat, value in topic_data[metric].items():
                merged_data[cat] += value

        categories = list(merged_data.keys())
        merged_values = [merged_data[cat] for cat in categories]
        nvd_values = [nvd_counts[metric][cat] for cat in categories]

        # Convert to percentages
        total_merged = sum(merged_values)
        total_nvd = sum(nvd_values)
        merged_percentages = [v / total_merged * 100 for v in merged_values]
        nvd_percentages = [v / total_nvd * 100 for v in nvd_values]

        x = np.arange(len(categories))
        width = 0.35

        ax = axs[idx, 0]
        rects1 = ax.bar(
            x - width / 2,
            merged_percentages,
            width,
            label=f"Top {k} Topics",
            color="skyblue",
        )
        rects2 = ax.bar(
            x + width / 2, nvd_percentages, width, label="NVD Data", color="orange"
        )

        ax.set_ylabel("Percentage")
        ax.set_title(f"{metric} - Merged Top {k} Topics vs NVD Data")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()

        # Add value labels on the bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

        autolabel(rects1)
        autolabel(rects2)

        # Add topic numbers and top words
        topic_info = f"Top {k} Topics: " + ", ".join(
            [f"{topic}" for topic, _ in top_k_topics]
        )
        topic_words = set()
        for _, topic_data in top_k_topics:
            topic_words.update(
                topic_data["topic_words"][:5]
            )  # Top 5 words from each topic
        topic_words_text = f"Common Topic Words:\n" + ", ".join(
            list(topic_words)[:15]
        )  # Limit to 15 words
        ax.text(
            0.5,
            -0.15,
            topic_info + "\n" + topic_words_text,
            ha="center",
            va="center",
            transform=ax.transAxes,
            wrap=True,
        )

    plt.tight_layout()
    plt.savefig(f"{path}_k{k}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved as {path}_k{k}.png")


def plot_best_fit_topic(
    topic_counts,
    nvd_counts,
    metric,
    category,
    path="./temp_plots/best_fit_topic_relative",
):
    # Find the topic with the highest percentage for the given metric and category
    best_topic = max(
        topic_counts.items(),
        key=lambda x: x[1][metric][category] / sum(x[1][metric].values()),
    )

    best_topic_num, best_topic_data = best_topic

    # Calculate percentages for the best topic
    best_topic_percentages = {
        cat: value / sum(nvd_counts[metric].values()) * 100
        for cat, value in best_topic_data[metric].items()
    }

    # Calculate percentages for NVD data
    nvd_percentages = {
        cat: value / sum(nvd_counts[metric].values()) * 100
        for cat, value in nvd_counts[metric].items()
    }

    # Prepare data for plotting
    categories = list(best_topic_percentages.keys())
    best_topic_values = [best_topic_percentages[cat] for cat in categories]
    nvd_values = [nvd_percentages[cat] for cat in categories]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax.bar(
        x - width / 2,
        best_topic_values,
        width,
        label=f"Topic {best_topic_num}",
        color="skyblue",
    )
    rects2 = ax.bar(x + width / 2, nvd_values, width, label="NVD Data", color="orange")

    ax.set_ylabel("Percentage")
    ax.set_title(f"Comparison of {metric} - Best Fit Topic for {category} vs NVD Data")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Add value labels on the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)

    # Add topic words as text
    topic_words_text = f"Top Topic Words (Topic {best_topic_num}):\n" + ", ".join(
        best_topic_data["topic_words"][:10]
    )
    plt.text(
        0.5,
        -0.15,
        topic_words_text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        wrap=True,
    )

    plt.tight_layout()
    plt.savefig(f"{path}_{metric}_{category}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved as {path}_{metric}_{category}.png")


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
