import numpy as np

metrics_and_dimensions = {
    "attackVector": ["NETWORK", "ADJACENT", "LOCAL", "PHYSICAL"],
    "attackComplexity": ["LOW", "HIGH"],
    "privilegesRequired": ["NONE", "LOW", "HIGH"],
    "userInteraction": ["NONE", "REQUIRED"],
    "scope": ["UNCHANGED", "CHANGED"],
    "confidentialityImpact": ["NONE", "LOW", "HIGH"],
    "integrityImpact": ["NONE", "LOW", "HIGH"],
    "availabilityImpact": ["NONE", "LOW", "HIGH"],
}
# Example usage for sampling initial probabilities

metrics_counts = {
    "attackVector": [
        {"NETWORK": []},
        {"ADJACENT": []},
        {"LOCAL": []},
        {"PHYSICAL": []},
    ],
    "attackComplexity": [{"LOW": []}, {"HIGH": []}],
    "privilegesRequired": [{"NONE": []}, {"LOW": []}, {"HIGH": []}],
    "userInteraction": [{"NONE": []}, {"REQUIRED": []}],
    "scope": [{"UNCHANGED": []}, {"CHANGED": []}],
    "confidentialityImpact": [{"NONE": []}, {"LOW": []}, {"HIGH": []}],
    "integrityImpact": [{"NONE": []}, {"LOW": []}, {"HIGH": []}],
    "availabilityImpact": [{"NONE": []}, {"LOW": []}, {"HIGH": []}],
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
# metric_probabilities = each metric distribution sumed to 1

# x is the score assigned for each metric
# for cve in cves:
#     for metric in metrics:
#         pass
