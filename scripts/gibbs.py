import numpy as np
import utils
import pymc3 as pm
from scipy.stats import dirichlet
import numpy as np
import metric_counts
from fastprogress import fastprogress
fastprogress.printing = lambda: True

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

confusion_matrices = {}
for metric in metrics_and_dimensions.keys():
    confusion_matrices[metric] = []


for metric in confusion_matrices.keys():
    for dimension in metrics_and_dimensions[metric]:
        new = []
        for dim in metrics_and_dimensions[metric]:
            new.append((dimension, dim))
        confusion_matrices[metric].append(new)




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

nvd_data = utils.read_data("../data/nvd_cleaned.pkl")
nvd_metrics = metric_counts.calculate_metric_counts(nvd_data["data"])
print(nvd_metrics)

sampled_probabilities = {k: np.random.dirichlet(v) for k, v in priors.items()}
# metric_probabilities = each metric distribution summed to 1


# Initial uniform priors for three levels of availability impact

# Observed totals from your data for 'availabilityImpact'
priors = []

# Update priors with observed data
for metric in nvd_metrics:
    priors.append(list(nvd_metrics[metric].values()))


print(priors)
exit()




alpha_posterior = [a + o for a, o in zip(alpha_prior, observed_totals)]

# Sample from the posterior distribution to get the expected probabilities
posterior_samples = dirichlet(alpha_posterior).rvs(size=1000)

# Calculate expected probabilities for each impact level
expected_probabilities = posterior_samples.mean(axis=0)
print("Expected Probabilities for 'Availability Impact':", expected_probabilities)
# mitre_metrics = metric_counts.calculate_metric_counts(mitre_overlap)
# mitre_data = utils.read_data("../data/mitre_cleaned.pkl")





# # Model setup in PyMC3
# with pm.Model() as model:
#     # Priors for categorical probabilities
#     category_probabilities = pm.Dirichlet('category_probabilities', a=np.array([1, 1, 1]))

#     # Confusion matrix as a set of Dirichlet distributions
#     confusion_matrix = pm.Dirichlet('confusion_matrix', 
#                                     a=np.array([[3, 1, 1],
#                                                 [1, 3, 1],
#                                                 [1, 1, 3]]),
#                                     shape=(3, 3))

#     # Likelihood (Multinomial here could be replaced depending on your data structure)
#     observed_data = pm.Multinomial('observed_data', n=np.sum(data), p=confusion_matrix[0], observed=data)

#     # Gibbs sampling using Metropolis-Hastings (default for discrete variables in PyMC3)
#     trace = pm.sample(1000, return_inferencedata=True, step=pm.Metropolis())

# # Extract sampled values (post burn-in)
# sampled_confusion_matrix = trace['confusion_matrix'][200:]  # Discard first 200 for burn-in

# # Display results
# print("Sampled Confusion Matrix Post Burn-in:")
# print(sampled_confusion_matrix.mean(axis=0))

