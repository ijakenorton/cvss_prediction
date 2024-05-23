import numpy as np
import utils

metrics_and_dimensions = {
    "attackVector": ["NETWORK", "ADJACENT_NETWORK", "LOCAL", "PHYSICAL"],
    "attackComplexity": ["LOW", "HIGH"],
    "privilegesRequired": ["NONE", "LOW", "HIGH"],
    "userInteraction": ["NONE", "REQUIRED"],
    "scope": ["UNCHANGED", "CHANGED"],
    "confidentialityImpact": ["NONE", "LOW", "HIGH"],
    "integrityImpact": ["NONE", "LOW", "HIGH"],
    "availabilityImpact": ["NONE", "LOW", "HIGH"],
}
# Example usage for sampling initial probabilities


def metrics_counts():
    return {
        "attackVector": {
            "NETWORK": 0,
            "ADJACENT_NETWORK": 0,
            "LOCAL": 0,
            "PHYSICAL": 0,
        },
        "attackComplexity": {
            "LOW": 0,
            "HIGH": 0,
        },
        "privilegesRequired": {
            "NONE": 0,
            "LOW": 0,
            "HIGH": 0,
        },
        "userInteraction": {
            "NONE": 0,
            "REQUIRED": 0,
        },
        "scope": {
            "UNCHANGED": 0,
            "CHANGED": 0,
        },
        "confidentialityImpact": {
            "NONE": 0,
            "LOW": 0,
            "HIGH": 0,
        },
        "integrityImpact": {
            "NONE": 0,
            "LOW": 0,
            "HIGH": 0,
        },
        "availabilityImpact": {
            "NONE": 0,
            "LOW": 0,
            "HIGH": 0,
        },
    }


confusion_matrices = {}
for metric in metrics_and_dimensions.keys():
    confusion_matrices[metric] = []


for metric in confusion_matrices.keys():
    for dimension in metrics_and_dimensions[metric]:
        new = []
        for dim in metrics_and_dimensions[metric]:
            new.append((dimension, dim))
        confusion_matrices[metric].append(new)


# for key in confusion_matrices:
#     print(key)
#     print("------------------------------------")
#     for line in confusion_matrices[key]:
#         print(line)


# Initialize priors and confusion matrices
priors = {}
confusion_matrices = {}

for metric, categories in metrics_and_dimensions.items():
    num_categories = len(categories)
    # Initialize Dirichlet priors with 1's for each category
    priors[metric] = np.array([1] * num_categories)
    # Initialize confusion matrix as a ones matrix
    confusion_matrices[metric] = np.ones((num_categories, num_categories), dtype=int)
    index = 0
    # Update priors to favour the database being correct
    for i in range(len(confusion_matrices[metric])):
        confusion_matrices[metric][i][index] = len(confusion_matrices[metric])
        index += 1


# Example usage for sampling initial probabilities and viewing matrices
sampled_probabilities = {k: np.random.dirichlet(v) for k, v in priors.items()}
print("Sampled Probabilities:", sampled_probabilities)
print("Confusion Matrices:", confusion_matrices)


# sampled_probabilities = {k: np.random.dirichlet(v) for k, v in priors.items()}
# metric_probabilities = each metric distribution summed to 1

nvd_data = utils.read_data("../data/nvd_cleaned.pkl")
mitre_data = utils.read_data("../data/mitre_cleaned.pkl")
# print(nvd_data["data"][0].keys())
# mitre_data = utils.read_data()

# for cve in nvd_data["data"]:
#     for metric, value in dict.items(cve["cvssData"]):
#         if metric == "baseScore":
#             continue

#         if value == "ADJACENT_NETWORK":
#             value = "ADJACENT"

#         metrics_counts[metric][value] += 1


# for key in metrics_counts:
#     print(metrics_counts[key])

# print("adjacent:", adjacent)

adjacent = 0


# x is the score assigned for each metric
