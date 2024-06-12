import matplotlib.pyplot as plt
import numpy as np
import pyjags


data_all_categories = {
    "N_attackVector": sum([80331, 2542, 26224, 1147]),
    "obs_attackVector": [80331, 2542, 26224, 1147],
    "alpha_attackVector": [1.0, 1.0, 1.0, 1.0],
    "N_attackComplexity": sum([104921, 5323]),
    "obs_attackComplexity": [104921, 5323],
    "alpha_attackComplexity": [1.0, 1.0],
    "N_privilegesRequired": sum([65195, 34306, 10743]),
    "obs_privilegesRequired": [65195, 34306, 10743],
    "alpha_privilegesRequired": [1.0, 1.0, 1.0],
    "N_userInteraction": sum([73571, 36673]),
    "obs_userInteraction": [73571, 36673],
    "alpha_userInteraction": [1.0, 1.0],
    "N_scope": sum([90349, 19895]),
    "obs_scope": [90349, 19895],
    "alpha_scope": [1.0, 1.0],
    "N_confidentialityImpact": sum([23683, 22864, 63697]),
    "obs_confidentialityImpact": [23683, 22864, 63697],
    "alpha_confidentialityImpact": [1.0, 1.0, 1.0],
    "N_integrityImpact": sum([34202, 21858, 54184]),
    "obs_integrityImpact": [34202, 21858, 54184],
    "alpha_integrityImpact": [1.0, 1.0, 1.0],
    "N_availabilityImpact": sum([44572, 2534, 63138]),
    "obs_availabilityImpact": [44572, 2534, 63138],
    "alpha_availabilityImpact": [1.0, 1.0, 1.0],
}


model_code = """
model {
    pi_attackVector ~ ddirch(alpha_attackVector);
    obs_attackVector ~ dmulti(pi_attackVector, N_attackVector);
    
    pi_attackComplexity ~ ddirch(alpha_attackComplexity);
    obs_attackComplexity ~ dmulti(pi_attackComplexity, N_attackComplexity);
    
    pi_privilegesRequired ~ ddirch(alpha_privilegesRequired);
    obs_privilegesRequired ~ dmulti(pi_privilegesRequired, N_privilegesRequired);
    
    pi_userInteraction ~ ddirch(alpha_userInteraction);
    obs_userInteraction ~ dmulti(pi_userInteraction, N_userInteraction);
    
    pi_scope ~ ddirch(alpha_scope);
    obs_scope ~ dmulti(pi_scope, N_scope);
    
    pi_confidentialityImpact ~ ddirch(alpha_confidentialityImpact);
    obs_confidentialityImpact ~ dmulti(pi_confidentialityImpact, N_confidentialityImpact);
    
    pi_integrityImpact ~ ddirch(alpha_integrityImpact);
    obs_integrityImpact ~ dmulti(pi_integrityImpact, N_integrityImpact);
    
    pi_availabilityImpact ~ ddirch(alpha_availabilityImpact);
    obs_availabilityImpact ~ dmulti(pi_availabilityImpact, N_availabilityImpact);
}
"""

# Create and run the model
# Initialize and run the model
model = pyjags.Model(
    code=model_code,
    data=data_all_categories,
    init=None,  # Automatic initialization
    chains=4,
    adapt=500,
    progress_bar=True,
)

# Burn-in phase
model.update(500)  # Burn-in period

# Sampling
samples = model.sample(
    1000,
    vars=[
        "pi_attackVector",
        "pi_attackComplexity",
        "pi_privilegesRequired",
        "pi_userInteraction",
        "pi_scope",
        "pi_confidentialityImpact",
        "pi_integrityImpact",
        "pi_availabilityImpact",
    ],
)

# Example visualization for one of the categories


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


for key, value in samples.items():
    plot_category(value, key, f"Posterior Distribution for {key}")
