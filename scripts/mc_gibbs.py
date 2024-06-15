import numpy as np
import pymc3 as pm
import theano.tensor as tt
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint


def plot_diagnostics(trace):
    summary = az.summary(trace)
    az.plot_trace(trace)
    plt.show()

    az.plot_autocorr(trace)
    plt.show()

    az.plot_posterior(trace)

    plt.show()


def plot_confusion_matrix(trace, var_name, categories, title="Confusion Matrix"):
    cm_mean = np.mean(
        trace.posterior[var_name].values, axis=(0, 1)
    )  # Average over chains and draws

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_mean,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories,
    )
    plt.xlabel("Predicted Category")
    plt.ylabel("True Category")
    plt.title(title)
    plt.show()


categories = ["NETWORK", "ADJACENT_NETWORK", "LOCAL", "PHYSICAL"]
mitre_counts = np.array([24202, 1854, 7100, 393])
nvd_counts = np.array([80331, 2542, 26224, 1147])
category_index = np.arange(len(categories))  # Category indices for the model

with pm.Model() as model:
    # Priors for the true category probabilities
    psi_attackVector = pm.Dirichlet("psi_attackVector", a=np.ones(len(categories)))

    # Confusion matrices for MITRE and NVD
    alpha_matrix = (
        np.ones((len(categories), len(categories))) + np.eye(len(categories)) * 2
    )  # Slight diagonal dominance
    pi_mitre = pm.Dirichlet(
        "pi_mitre",
        a=alpha_matrix,
        shape=(len(categories), len(categories)),
    )
    pi_nvd = pm.Dirichlet(
        "pi_nvd",
        a=alpha_matrix,
        shape=(len(categories), len(categories)),
    )

    # Compute expected probabilities for each database
    expected_mitre = tt.dot(psi_attackVector, pi_mitre)
    expected_nvd = tt.dot(psi_attackVector, pi_nvd)

    # Likelihood for observed count data using Multinomial
    obs_mitre = pm.Multinomial(
        "obs_mitre", n=sum(mitre_counts), p=expected_mitre, observed=mitre_counts
    )
    obs_nvd = pm.Multinomial(
        "obs_nvd", n=sum(nvd_counts), p=expected_nvd, observed=nvd_counts
    )

    # Sampling
    pm.set_tt_rng(42)
    trace = pm.sample(
        2000, chains=8, tune=1500, target_accept=0.95, return_inferencedata=True
    )
    # pprint(trace)
    # plot_diagnostics(trace)
    # Plot confusion matrix for MITRE
    plot_confusion_matrix(
        trace, "pi_mitre", categories, title="Confusion Matrix for MITRE"
    )

    # Plot confusion matrix for NVD
    plot_confusion_matrix(trace, "pi_nvd", categories, title="Confusion Matrix for NVD")

    # Accessing the trace for pi_mitre directly
    pi_mitre_samples = trace.posterior["pi_mitre"].values

    # Accessing a specific element, e.g., the probability that the true category "NETWORK" is predicted as "LOCAL"
    network_as_local_mitre = pi_mitre_samples[
        :, :, 0, 2
    ]  # Assuming "NETWORK" is index 0 and "LOCAL" is index 2

    # Plot the distribution of this element
    sns.histplot(network_as_local_mitre.flatten(), kde=True)
    plt.title("Posterior distribution for NETWORK predicted as LOCAL in MITRE")
    plt.xlabel("Probability")
    plt.ylabel("Density")
    plt.show()
    # az.plot_trace(trace)
    # az.plot_pair(
    #     trace,
    #     var_names=["psi_attackVector", "pi_mitre", "pi_nvd"],
    #     kind="scatter",
    #     divergences=True,
    # )
    ppc = pm.sample_posterior_predictive(trace, var_names=["obs_mitre", "obs_nvd"])
    # Compare observed counts to posterior predictive counts
    _, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(
        ppc["obs_mitre"][:, 0],
        kde=True,
        color="skyblue",
        ax=ax,
        label="Posterior Predictive NETWORK Counts - MITRE",
    )
    ax.axvline(
        mitre_counts[0],
        color="red",
        linestyle="dashed",
        linewidth=2,
        label="Observed NETWORK Count - MITRE",
    )
    ax.legend()
    plt.title("Posterior Predictive Check for NETWORK - MITRE")
    plt.show()

    # summary = az.summary(trace, var_names=["pi_mitre", "pi_nvd"])
    summary = az.summary(trace)
    print(summary)
