import matplotlib.pyplot as plt
import numpy as np
import pyjags
from cvss_types import Metrics
import metric_counts
import utils
import overlap
from pprint import pprint

mitre = {
    "attackVector": {
        "NETWORK": 24202,
        "ADJACENT_NETWORK": 1854,
        "LOCAL": 7100,
        "PHYSICAL": 393,
    },
    "attackComplexity": {"LOW": 28179, "HIGH": 5370},
    "privilegesRequired": {"NONE": 15684, "LOW": 12294, "HIGH": 5571},
    "userInteraction": {"NONE": 21963, "REQUIRED": 11586},
    "scope": {"UNCHANGED": 24530, "CHANGED": 9019},
    "confidentialityImpact": {"NONE": 9215, "LOW": 10474, "HIGH": 13860},
    "integrityImpact": {"NONE": 10032, "LOW": 11898, "HIGH": 11619},
    "availabilityImpact": {"NONE": 13550, "LOW": 7348, "HIGH": 12651},
}

nvd = {
    "attackVector": {
        "NETWORK": 80331,
        "ADJACENT_NETWORK": 2542,
        "LOCAL": 26224,
        "PHYSICAL": 1147,
    },
    "attackComplexity": {"LOW": 104921, "HIGH": 5323},
    "privilegesRequired": {"NONE": 65195, "LOW": 34306, "HIGH": 10743},
    "userInteraction": {"NONE": 73571, "REQUIRED": 36673},
    "scope": {"UNCHANGED": 90349, "CHANGED": 19895},
    "confidentialityImpact": {"NONE": 23683, "LOW": 22864, "HIGH": 63697},
    "integrityImpact": {"NONE": 34202, "LOW": 21858, "HIGH": 54184},
    "availabilityImpact": {"NONE": 44572, "LOW": 2534, "HIGH": 63138},
}

category_data = {
    "alpha_attackVector": [
        1.0,
        1.0,
        1.0,
        1.0,
    ],  # Assuming a uniform prior for simplicity
    "N_attackVector": sum([24202, 1854, 7100, 393]) + sum([80331, 2542, 26224, 1147]),
    "obs_mitre_attackVector": [24202, 1854, 7100, 393],
    "obs_nvd_attackVector": [80331, 2542, 26224, 1147],
}
category_data["obs_mitre_attackVector"] = list(category_data["obs_mitre_attackVector"])
category_data["obs_nvd_attackVector"] = list(category_data["obs_nvd_attackVector"])


model_code_attackVector = """
model {
    psi_attackVector ~ ddirch(alpha_attackVector);  # Dirichlet prior for true score probabilities

    # Confusion matrices for MITRE and NVD
    for (i in 1:4) {
        for (j in 1:4) {
            pi_mitre_attackVector[i, j] ~ dbeta(1, 1);  # Probabilities for MITRE
            pi_nvd_attackVector[i, j] ~ dbeta(1, 1);   # Probabilities for NVD
        }
    }

    # True scores and observed scores for each category
    for (j in 1:4) {
        true_score_attackVector[j] ~ dcat(psi_attackVector[]);  # Draw from Dirichlet distribution

        # Directly use probability vectors from the confusion matrices in the dcat function
        obs_mitre_attackVector[j] ~ dcat(pi_mitre_attackVector[true_score_attackVector[j], ]);
        obs_nvd_attackVector[j] ~ dcat(pi_nvd_attackVector[true_score_attackVector[j], ]);
    }
}
"""

# Data for running the model
jags_data_attackVector = {
    "alpha_attackVector": category_data["alpha_attackVector"],
    "obs_mitre_attackVector": category_data["obs_mitre_attackVector"],
    "obs_nvd_attackVector": category_data["obs_nvd_attackVector"],
}
jags_data_attackVector = {
    "alpha_attackVector": [1.0, 1.0, 1.0, 1.0],  # Uniform priors for the 4 categories
    "obs_mitre_attackVector": [24202, 1854, 7100, 393],  # Observations from MITRE
    "obs_nvd_attackVector": [80331, 2542, 26224, 1147],  # Observations from NVD
}

# Specify which variables to monitor
model = pyjags.Model(
    code=model_code_attackVector,
    data=jags_data_attackVector,
    chains=3,
    adapt=500,
    progress_bar=True,
)
model.update(500)  # Burn-in period
samples = model.sample(
    1000,
    vars=[
        "psi_attackVector",
        "pi_mitre_attackVector",
        "pi_nvd_attackVector",
        "obs_nvd_attackVector",
        "obs_mitre_attackVector",
    ],
)


# You can analyze or visualize your samples here


def plot_category(samples, category_name, title):
    """
    This function plots histograms for the given category's parameter samples.
    `samples` is a numpy array with shape (chains, iterations, parameters).
    `category_name` is the name used for labeling.
    `title` is the title of the histogram.
    """
    # Reshape samples to collapse chains and iterations
    reshaped_samples = samples.reshape(-1, samples.shape[-1])

    # Create a figure
    plt.figure(figsize=(12, 6))

    # Plot histogram for each parameter in the category
    for i in range(reshaped_samples.shape[1]):
        plt.hist(
            reshaped_samples[:, i],
            # bins=50,
            alpha=0.6,
            label=f"{category_name} Param {i+1}",
        )

    plt.title(title)
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
