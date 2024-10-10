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
                [0.73, 0.04, 0.22, 0.01],
                [0.17, 0.52, 0.17, 0.14],
                [0.19, 0.17, 0.55, 0.10],
                [0.18, 0.18, 0.18, 0.47],
            ]
        ),
        "labels": (["AN", "L", "N", "P"], ["AN", "L", "N", "P"]),
    },
    "Attack Complexity": {
        "data": np.array([[0.87, 0.13], [0.25, 0.75]]),
        "labels": (["L", "H"], ["L", "H"]),
    },
    "Privileges Required": {
        "data": np.array([[0.65, 0.24, 0.11], [0.21, 0.64, 0.15], [0.21, 0.21, 0.58]]),
        "labels": (["N", "L", "H"], ["N", "L", "H"]),
    },
    "User Interaction": {
        "data": np.array([[0.75, 0.25], [0.25, 0.75]]),
        "labels": (["N", "R"], ["N", "R"]),
    },
    "Scope": {
        "data": np.array([[0.85, 0.15], [0.26, 0.74]]),
        "labels": (["U", "C"], ["U", "C"]),
    },
    "Confidentiality Impact": {
        "data": np.array([[0.59, 0.21, 0.21], [0.19, 0.60, 0.21], [0.17, 0.19, 0.64]]),
        "labels": (["N", "L", "H"], ["N", "L", "H"]),
    },
    "Integrity Impact": {
        "data": np.array([[0.57, 0.23, 0.20], [0.19, 0.61, 0.20], [0.18, 0.23, 0.59]]),
        "labels": (["N", "L", "H"], ["N", "L", "H"]),
    },
    "Availability Impact": {
        "data": np.array([[0.60, 0.19, 0.20], [0.20, 0.60, 0.20], [0.23, 0.19, 0.58]]),
        "labels": (["N", "L", "H"], ["N", "L", "H"]),
    },
}

# Generate and save heatmaps
for title, matrix in matrices.items():
    create_heatmap(
        matrix["data"],
        matrix["labels"],
        "",
        f"./new_gibbs/{title.lower().replace(' ', '_')}_mitre.pdf",
    )

print("All heatmaps have been generated and saved.")
