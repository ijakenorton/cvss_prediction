import matplotlib.pyplot as plt


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
