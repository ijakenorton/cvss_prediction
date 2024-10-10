import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Function to create and save a heatmap
def create_heatmap(data, labels, title, filename):
    plt.figure(figsize=(len(labels[1]) * 0.8, len(labels[0]) * 0.8))
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=labels[1],
        yticklabels=labels[0],
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# Data for each confusion matrix
matrices = {
    "Attack Vector": {
        "data": np.array(
            [
                [0.77, 0.01, 0.22, 0.01],
                [0.19, 0.49, 0.19, 0.13],
                [0.20, 0.13, 0.57, 0.10],
                [0.18, 0.18, 0.18, 0.46],
            ]
        ),
        "labels": (["AN", "L", "N", "P"], ["AN", "L", "N", "P"]),
    },
    "Attack Complexity": {
        "data": np.array([[0.98, 0.02], [0.34, 0.66]]),
        "labels": (["L", "H"], ["L", "H"]),
    },
    "Privileges Required": {
        "data": np.array([[0.78, 0.16, 0.06], [0.29, 0.61, 0.10], [0.25, 0.23, 0.52]]),
        "labels": (["N", "L", "H"], ["N", "L", "H"]),
    },
    "User Interaction": {
        "data": np.array([[0.82, 0.18], [0.33, 0.67]]),
        "labels": (["N", "R"], ["N", "R"]),
    },
    "Scope": {
        "data": np.array([[0.91, 0.09], [0.32, 0.68]]),
        "labels": (["U", "C"], ["U", "C"]),
    },
    "Confidentiality Impact": {
        "data": np.array([[0.55, 0.18, 0.27], [0.19, 0.53, 0.28], [0.12, 0.11, 0.77]]),
        "labels": (["N", "L", "H"], ["N", "L", "H"]),
    },
    "Integrity Impact": {
        "data": np.array([[0.60, 0.15, 0.25], [0.23, 0.50, 0.27], [0.17, 0.11, 0.71]]),
        "labels": (["N", "L", "H"], ["N", "L", "H"]),
    },
    "Availability Impact": {
        "data": np.array([[0.68, 0.03, 0.29], [0.26, 0.48, 0.26], [0.22, 0.02, 0.77]]),
        "labels": (["N", "L", "H"], ["N", "L", "H"]),
    },
}

# Generate and save heatmaps
for title, matrix in matrices.items():
    create_heatmap(
        matrix["data"],
        matrix["labels"],
        "",
        f"./new_gibbs/{title.lower().replace(' ', '_')}_nvd.pdf",
    )

print("All heatmaps have been generated and saved.")
