import matplotlib.pyplot as plt

# Updated data from the user's new dictionaries
# mitre = {
#     "attackVector": {
#         "NETWORK": 24202,
#         "ADJACENT_NETWORK": 1854,
#         "LOCAL": 7100,
#         "PHYSICAL": 393,
#     },
#     "attackComplexity": {"LOW": 28179, "HIGH": 5370},
#     "privilegesRequired": {"NONE": 15684, "LOW": 12294, "HIGH": 5571},
#     "userInteraction": {"NONE": 21963, "REQUIRED": 11586},
#     "scope": {"UNCHANGED": 24530, "CHANGED": 9019},
#     "confidentialityImpact": {"NONE": 9215, "LOW": 10474, "HIGH": 13860},
#     "integrityImpact": {"NONE": 10032, "LOW": 11898, "HIGH": 11619},
#     "availabilityImpact": {"NONE": 13550, "LOW": 7348, "HIGH": 12651},
# }


# nvd = {
#     "attackVector": {
#         "NETWORK": 24202,
#         "ADJACENT_NETWORK": 1854,
#         "LOCAL": 7100,
#         "PHYSICAL": 393,
#     },
#     "attackComplexity": {"LOW": 28179, "HIGH": 5370},
#     "privilegesRequired": {"NONE": 15684, "LOW": 12294, "HIGH": 5571},
#     "userInteraction": {"NONE": 21963, "REQUIRED": 11586},
#     "scope": {"UNCHANGED": 24530, "CHANGED": 9019},
#     "confidentialityImpact": {"NONE": 9215, "LOW": 10474, "HIGH": 13860},
#     "integrityImpact": {"NONE": 10032, "LOW": 11898, "HIGH": 11619},
#     "availabilityImpact": {"NONE": 13550, "LOW": 7348, "HIGH": 12651},
# }
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
