from enum import Enum
from multiprocessing.sharedctypes import Value

from numpy.distutils.system_info import NotFoundError
import temp_data
from typing import Dict
import numpy as np
import argparse
from cvss_types import ConfusionMatrixInfo, Metrics
import confusion_generator
import utils
import pymc3 as pm
import overlap
from scipy.stats import dirichlet
import numpy as np
import metric_counts
from fastprogress import fastprogress
import arviz as az
import matplotlib.pyplot as plt

fastprogress.printing = lambda: True
pass
DATA_TYPES = ["mitre", "nvd", "combined", "overlap", "all"]
DATA_NAME = ""

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
    metrics = counts(read_data())

    metrics_observed = {
        metric: np.array(list(data.values())) for metric, data in metrics.items()
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
                500, tune=2000, return_inferencedata=False, step=pm.Metropolis()
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

    with open(f"../results/{DATA_NAME}_metropolis", "w") as file:
        for metric, result in results.items():
            file.write(f"Results for {metric}:\n")
            file.write(f"Probabilities: result['probabilities']\n")
            file.write(f"Confusion Matrix:\n {result['confusion_matrix']}\n")
            file.write("\n")


def compute_gibbs():
    metrics = counts(read_data())

    metrics_observed = {
        metric: np.array(list(data.values())) for metric, data in metrics.items()
    }

    results = {}

    for metric, observed_totals in metrics_observed.items():
        with pm.Model() as model:
            num_categories = len(observed_totals)
            # Prior for the category probabilities
            theta = pm.Dirichlet(
                "theta",
                a=np.ones(num_categories) + observed_totals,
                shape=num_categories,
            )

            # Confusion matrix as a set of Dirichlet distributions
            alpha_matrix = np.ones((num_categories, num_categories)) + np.eye(
                num_categories
            )
            confusion_matrix = pm.Dirichlet(
                "confusion_matrix",
                a=alpha_matrix,
                shape=(num_categories, num_categories),
            )

            # Define categorical outcomes influenced by the confusion matrix
            categories = [
                pm.Categorical(f"category_{i}", p=confusion_matrix[i, :])
                for i in range(num_categories)
            ]

            # Use CategoricalGibbsMetropolis for the categories
            step_cat = [pm.CategoricalGibbsMetropolis(vars=[cat]) for cat in categories]

            # Sampling
            trace = pm.sample(
                500,
                tune=2000,
                step=step_cat,
                return_inferencedata=False,
                chains=8,
                cores=8,
            )

            # Post-processing
            burn_in = 250
            sampled_theta = trace.get_values("theta", burn=burn_in, combine=True).mean(
                axis=0
            )
            sampled_confusion_matrix = trace.get_values(
                "confusion_matrix", burn=burn_in, combine=True
            ).mean(axis=0)

            results[metric] = {
                "theta": sampled_theta,
                "confusion_matrix": sampled_confusion_matrix,
            }

    infos = []
    with open(f"../results/{DATA_NAME}_gibbs_formatted", "w") as file:
        for metric, result in results.items():

            theta = list(map(lambda x: f"{x:.2g}\t", result["theta"]))
            file.write(f"Results for {metric}:\n")
            file.write(f"Theta (Category Probabilities): {theta}\n")
            file.write(f"Confusion Matrix:\n {result['confusion_matrix']}\n")
            file.write("\n")

            inverse = utils.inverse_from_keys(list(metrics[metric].keys()))
            inverse = list(map(lambda x: f"{metric} {x}", inverse))
            info: ConfusionMatrixInfo = {
                "columns": inverse,
                "caption": f"Confusion matrices {metric}",
                "label": f"table:{DATA_NAME}-{metric}",
                "row_labels": inverse,
                "data": result["confusion_matrix"],
            }
            infos.append(info)

    return infos


def compute_bayes():
    metrics = counts(read_data())
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
        if metric in metrics:
            observed = np.array(list(metrics[metric].values()))
            observed_totals.append(observed)
        else:
            # Initialize with zeros if no data is available for this metric
            observed_totals.append(np.zeros(len(metrics_and_dimensions[metric])))

    # Calculate updated probabilities and print them
    with open(f"../results/{DATA_NAME}_bayes", "w") as file:
        for i, (prior, observed) in enumerate(zip(confusion_matrices, observed_totals)):
            updated_prior = prior.diagonal() + observed
            posterior_samples = dirichlet(updated_prior).rvs(size=1000)
            expected_probabilities = posterior_samples.mean(axis=0)
            file.write(
                f"Expected Probabilities for '{list(metrics_and_dimensions.keys())[i]}':{expected_probabilities}\n"
            )
            file.write("\n")


def read_data():
    match DATA_NAME:
        case "nvd" | "mitre":
            data = utils.read_data(f"../data/{DATA_NAME}_2.0_cleaned.pkl")
            if data is None:
                raise ValueError(f"Data not found for {DATA_NAME}")
            return {DATA_NAME: data["data"]}
        case "combined":
            nvd = utils.read_data(f"../data/nvd_cleaned.pkl")
            mitre = utils.read_data(f"../data/mitre_cleaned.pkl")
            if nvd is None or mitre is None:
                raise ValueError("NVD or MITRE data not found")
            return {"nvd": nvd["data"], "mitre": mitre["data"]}
        case "overlap":
            nvd, mitre = overlap.overlap()
            if nvd is None:
                raise ValueError(f"Data not found for nvd")
            if mitre is None:
                raise ValueError(f"Data not found for mitre")
            return {"nvd": nvd, "mitre": mitre}
        case _:
            raise ValueError("Dataname is incorrect")


def counts(data) -> Metrics:
    match DATA_NAME:
        case "nvd" | "mitre":
            counts: Metrics = metric_counts.calculate_metric_counts(data[DATA_NAME])
            return counts
        case "overlap" | "combined":
            nvd_counts: Metrics = metric_counts.calculate_metric_counts(data["nvd"])
            mitre_counts: Metrics = metric_counts.calculate_metric_counts(data["mitre"])
            for metric in nvd_counts:
                for answer in nvd_counts[metric]:
                    nvd_counts[metric][answer] += mitre_counts[metric][answer]
            return nvd_counts
        case _:
            raise ValueError("Dataname is incorrect")


def main():
    parser = argparse.ArgumentParser(
        description="Bayes modelling of CVE to CVSS scores"
    )

    parser.add_argument(
        "--source",
        choices=DATA_TYPES,
        required=True,
        help="The source to use (mitre or nvd or combined or overlap)",
    )

    # Parse the arguments
    args = parser.parse_args()
    global DATA_NAME
    DATA_NAME = args.source
    print(f"Using source: {args.source}")

    data = {}
    data["all"] = []
    if DATA_NAME == "all":
        for data_set in DATA_TYPES:
            DATA_NAME = data_set
            if data_set == "all":
                continue
            data[DATA_NAME] = compute_gibbs()

        # data = temp_data.data()
        for metric_idx in range(len(data[DATA_TYPES[0]])):
            row_labels = data["mitre"][metric_idx]["columns"]
            column_labels = []
            print(data.keys())
            for source in data.keys():
                if source == "all":
                    continue
                for item in data[source][metric_idx]["columns"]:
                    column_labels.append(item)

            data["all"].append(
                {
                    "columns": column_labels,
                    "caption": data[DATA_TYPES[0]][metric_idx]["caption"],
                    "label": data[DATA_TYPES[0]][metric_idx]["label"],
                    "row_labels": row_labels,
                    "data": np.hstack(
                        (
                            data["nvd"][metric_idx]["data"],
                            data["mitre"][metric_idx]["data"],
                            data["combined"][metric_idx]["data"],
                            data["overlap"][metric_idx]["data"],
                        )
                    ),
                }
            )

        with open("./table_data", "w") as file:
            file.write(str((data["all"])))
        for metric_idx in range(len(data["all"])):
            confusion_generator.generate(data["all"][metric_idx])

    else:
        infos = compute_metropolis_hastings()
        # for info in infos:
        #     confusion_generator.generate(info)


if __name__ == "__main__":
    main()
