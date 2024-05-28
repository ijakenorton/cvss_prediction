import numpy as np
import utils
import pymc3 as pm
from scipy.stats import dirichlet
import numpy as np
import metric_counts
from fastprogress import fastprogress
import arviz as az
import matplotlib.pyplot as plt

fastprogress.printing = lambda: True
pass

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


def plot_diagnostics(trace):
    summary = az.summary(trace)
    print(summary)
    # Trace plot for all variables
    az.plot_trace(trace)
    # Show the figure
    plt.show()

    # Autocorrelation plot for all variables
    az.plot_autocorr(trace)
    # Show the figure
    plt.show()

    # Posterior plot for all variables after burn-in
    az.plot_posterior(trace)

    # Show the figure
    plt.show()


def compute_metropolis_hastings():
    nvd_data = utils.read_data("../data/nvd_cleaned.pkl")
    nvd_metrics = metric_counts.calculate_metric_counts(nvd_data["data"])

    metrics_observed = {
        metric: np.array(list(data.values())) for metric, data in nvd_metrics.items()
    }

    results = {}

    for metric, observed_totals in metrics_observed.items():
        with pm.Model() as model:
            num_categories = len(observed_totals)

            category_probabilities = pm.Dirichlet(
                "category_probabilities", a=observed_totals + 1
            )

            alpha_matrix = (
                np.ones((num_categories, num_categories)) + np.eye(num_categories) * 2
            )  # Slight diagonal dominance

            pm.Dirichlet(
                "confusion_matrix",
                a=alpha_matrix,
                shape=(num_categories, num_categories),
            )

            # Likelihood
            pm.Multinomial(
                "observed_data",
                n=observed_totals.sum(),
                p=category_probabilities,
                observed=observed_totals,
            )

            # Sampling
            trace = pm.sample(
                1000, tune=1000, return_inferencedata=False, step=pm.Metropolis()
            )

            burn_in = 250
            sampled_category_probabilities = trace["category_probabilities"][
                burn_in:
            ].mean(axis=0)
            sampled_confusion_matrix = trace["confusion_matrix"][burn_in:].mean(axis=0)

            results[metric] = {
                "probabilities": sampled_category_probabilities,
                "confusion_matrix": sampled_confusion_matrix,
            }

    # Printing results for all metrics
    for metric, result in results.items():
        print(f"Results for {metric}:")
        print("Probabilities:", result["probabilities"])
        print("Confusion Matrix:\n", result["confusion_matrix"])
        print("\n")


def compute_gibbs():
    nvd_data = utils.read_data("../data/nvd_cleaned.pkl")
    nvd_metrics = metric_counts.calculate_metric_counts(nvd_data["data"])

    metrics_observed = {
        metric: np.array(list(data.values())) for metric, data in nvd_metrics.items()
    }

    results = {}

    for metric, observed_totals in metrics_observed.items():
        with pm.Model() as model:
            # Prior for the category probabilities
            theta = pm.Dirichlet(
                "theta", a=np.ones(len(observed_totals)) + observed_totals
            )

            # Categorical variables influenced by theta, not observed directly
            categories = [
                pm.Categorical(f"category_{i}", p=theta)
                for i in range(len(observed_totals))
            ]

            # Use CategoricalGibbsMetropolis for the categories
            step_cat = [pm.CategoricalGibbsMetropolis(vars=[cat]) for cat in categories]

            # Sampling
            trace = pm.sample(
                1000, tune=1000, step=step_cat, return_inferencedata=False
            )

            # Extract and store the results post burn-in
            burn_in = 250
            sampled_theta = trace.get_values("theta", burn=burn_in, combine=True).mean(
                axis=0
            )

            # plot_diagnostics(sampled_theta)
            results[metric] = {
                "theta": sampled_theta,
            }

    # Printing results for all metrics
    for metric, result in results.items():
        print(f"Results for {metric}:")
        print("Theta:", result["theta"])
        print("\n")


def compute_bayes():
    nvd_data = utils.read_data("../data/nvd_cleaned.pkl")
    nvd_metrics = metric_counts.calculate_metric_counts(nvd_data["data"])
    # Initialize priors and confusion matrices
    priors = []
    confusion_matrices = []

    for categories in metrics_and_dimensions.values():
        num_categories = len(categories)
        category_priors = np.ones(num_categories)
        confusion_matrix = np.eye(num_categories, dtype=int) * num_categories
        priors.append(category_priors)
        confusion_matrices.append(confusion_matrix)

    # Update priors with observed data
    observed_totals = []
    for metric in metrics_and_dimensions:
        if metric in nvd_metrics:
            observed = np.array(list(nvd_metrics[metric].values()))
            observed_totals.append(observed)
        else:
            # Initialize with zeros if no data is available for this metric
            observed_totals.append(np.zeros(len(metrics_and_dimensions[metric])))

    # Calculate updated probabilities and print them
    for i, (prior, observed) in enumerate(zip(confusion_matrices, observed_totals)):
        updated_prior = prior.diagonal() + observed
        posterior_samples = dirichlet(updated_prior).rvs(size=1000)
        expected_probabilities = posterior_samples.mean(axis=0)
        print(
            f"Expected Probabilities for '{list(metrics_and_dimensions.keys())[i]}': {expected_probabilities}"
        )


if __name__ == "__main__":
    compute_metropolis_hastings()
