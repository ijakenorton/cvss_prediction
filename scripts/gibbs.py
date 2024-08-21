from os.path import isfile
from pprint import pprint
import numpy as np
import pymc3 as pm
import theano.tensor as tt
import argparse
from cvss_types import ConfusionMatrixInfo, Metrics
import confusion_generator
import utils
import overlap
import parse_data

# from scipy.stats import dirichlet
import numpy as np
import metric_counts
import os

import seaborn
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

DATA_TYPES = ["mitre", "nvd", "combined", "overlap", "all"]
DATA_NAME = ""
CVSS_VERSION = "3.1"


def plot_confusion_matrix(
    trace,
    var_name,
    categories,
    title,
    figure,
):

    cm_mean = np.mean(trace.posterior[var_name].values, axis=(0, 1))

    figure.add_subplot(
        seaborn.heatmap(
            cm_mean,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=categories,
            yticklabels=categories,
        )
        # .set_xlabel("Predicted Category")
        # .set_ylabel("True Category")
    )
    # ax.set_xlabel("Predicted Category")
    # ax.set_ylabel("True Category")
    # ax.set_title(title)
    return figure


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

    nvd, mitre = counts(read_data())
    print(nvd, mitre)
    exit()

    nvd_observed = {
        metric: np.array(list(data.values())) for metric, data in nvd.items()
    }

    mitre_observed = {
        metric: np.array(list(data.values())) for metric, data in mitre.items()
    }

    if CVSS_VERSION == 2:
        nvd_fig, nvd_axes = plt.subplots(2, 3, figsize=(15, 15))
        mitre_fig, mitre_axes = plt.subplots(2, 3, figsize=(15, 15))
    else:
        nvd_fig, nvd_axes = plt.subplots(2, 4, figsize=(15, 15))
        mitre_fig, mitre_axes = plt.subplots(2, 4, figsize=(15, 15))

    for i, metric in enumerate(nvd_observed):

        nvd_counts = nvd_observed[metric]
        mitre_counts = mitre_observed[metric]
        categories = utils.dimensions(metric, CVSS_VERSION)
        with pm.Model() as model:
            num_categories = len(nvd_counts)

            # Priors for the true category probabilities
            psi_attackVector = pm.Dirichlet(f"psi_{metric}", a=np.ones(num_categories))

            # Confusion matrices for MITRE and NVD
            alpha_matrix = (
                np.ones((num_categories, num_categories)) + np.eye(num_categories) * 2
            )  # Slight diagonal dominance
            pi_mitre = pm.Dirichlet(
                "pi_mitre",
                a=alpha_matrix,
                shape=(num_categories, num_categories),
            )
            pi_nvd = pm.Dirichlet(
                "pi_nvd",
                a=alpha_matrix,
                shape=(num_categories, num_categories),
            )

            # Compute expected probabilities for each database
            expected_mitre = tt.dot(psi_attackVector, pi_mitre)
            expected_nvd = tt.dot(psi_attackVector, pi_nvd)

            # Likelihood for observed count data using Multinomial
            obs_mitre = pm.Multinomial(
                "obs_mitre",
                n=sum(mitre_counts),
                p=expected_mitre,
                observed=mitre_counts,
            )
            obs_nvd = pm.Multinomial(
                "obs_nvd", n=sum(nvd_counts), p=expected_nvd, observed=nvd_counts
            )

            # Sampling
            pm.set_tt_rng(42)
            trace = pm.sample(
                2000,
                chains=8,
                cores=8,
                tune=2000,
                target_accept=0.95,
                return_inferencedata=True,
            )
            # categories = utils.dimensions(metric, CVSS_VERSION)
            cm_mean = np.mean(trace.posterior["pi_nvd"].values, axis=(0, 1))

            seaborn.heatmap(
                cm_mean,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=categories,
                yticklabels=categories,
                ax=nvd_axes.flat[i],
            )
            cm_mean = np.mean(trace.posterior["pi_mitre"].values, axis=(0, 1))
            # Plot confusion matrix for NVD
            seaborn.heatmap(
                cm_mean,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=categories,
                yticklabels=categories,
                ax=mitre_axes.flat[i],
            )
    try:
        nvd_fig.savefig(f"../plots/nvd_{CVSS_VERSION}.png")
        mitre_fig.savefig(f"../plots/mitre_{CVSS_VERSION}.png")
    except Exception as e:
        print(f"Error: {e}")

    return None, None

    # with open(f"../results/{data_name}_metropolis", "w") as file:
    #     for metric, result in results.items():
    #         file.write(f"Results for {metric}:\n")
    #         file.write(f"Probabilities: result['probabilities']\n")
    #         file.write(f"Confusion Matrix:\n {result['confusion_matrix']}\n")
    #         file.write("\n")


# def compute_gibbs():
#     metrics = counts(read_data())

#     metrics_observed = {
#         metric: np.array(list(data.values())) for metric, data in metrics.items()
#     }

#     results = {}

#     for metric, observed_totals in metrics_observed.items():
#         with pm.Model() as model:
#             num_categories = len(observed_totals)
#             # Prior for the category probabilities
#             theta = pm.Dirichlet(
#                 "theta",
#                 a=np.ones(num_categories) + observed_totals,
#                 shape=num_categories,
#             )

#             # Use CategoricalGibbsMetropolis for the categories
#             step_cat = [pm.CategoricalGibbsMetropolis(vars=[cat]) for cat in categories]

#             # Sampling
#             trace = pm.sample(
#                 500,
#                 tune=2000,
#                 step=step_cat,
#                 return_inferencedata=False,
#                 chains=8,
#                 cores=8,
#             )

#             # Post-processing
#             burn_in = 250
#             sampled_theta = trace.get_values("theta", burn=burn_in, combine=True).mean(
#                 axis=0
#             )
#             sampled_confusion_matrix = trace.get_values(
#                 "confusion_matrix", burn=burn_in, combine=True
#             ).mean(axis=0)

#             results[metric] = {
#                 "theta": sampled_theta,
#                 "confusion_matrix": sampled_confusion_matrix,
#             }

#     infos = []
#     with open(f"../results/{DATA_NAME}_gibbs_formatted", "w") as file:
#         for metric, result in results.items():

#             theta = list(map(lambda x: f"{x:.2g}\t", result["theta"]))
#             file.write(f"Results for {metric}:\n")
#             file.write(f"Theta (Category Probabilities): {theta}\n")
#             file.write(f"Confusion Matrix:\n {result['confusion_matrix']}\n")
#             file.write("\n")

#             inverse = utils.inverse_from_keys(list(metrics[metric].keys()))
#             inverse = list(map(lambda x: f"{metric} {x}", inverse))
#             info: ConfusionMatrixInfo = {
#                 "columns": inverse,
#                 "caption": f"Confusion matrices {metric}",
#                 "label": f"table:{DATA_NAME}-{metric}",
#                 "row_labels": inverse,
#                 "data": result["confusion_matrix"],
#             }
#             infos.append(info)

#     return infos


def read_data():
    match DATA_NAME:
        case "nvd" | "mitre":
            data = utils.read_data(f"../data/{DATA_NAME}_{CVSS_VERSION}_cleaned.pkl")
            if data is None:
                raise ValueError(f"Data not found for {DATA_NAME}")
            return {DATA_NAME: data["data"]}
        case "combined":

            if os.path.isfile(f"../data/nvd_{CVSS_VERSION}_cleaned.pkl"):
                nvd = utils.read_data(f"../data/nvd_{CVSS_VERSION}_cleaned.pkl")
            else:
                # TODO fix this
                nvd = parse_data.parse("nvd", CVSS_VERSION)

            if os.path.isfile(f"../data/mitre_{CVSS_VERSION}_cleaned.pkl"):
                mitre = utils.read_data(f"../data/mitre_{CVSS_VERSION}_cleaned.pkl")
            else:
                # TODO fix this
                mitre = parse_data.parse("mitre", CVSS_VERSION)

            return {"nvd": nvd, "mitre": mitre}
        case "overlap":
            nvd, mitre = overlap.overlap()
            if nvd is None:
                raise ValueError(f"Data not found for nvd")
            if mitre is None:
                raise ValueError(f"Data not found for mitre")
            return {"nvd": nvd, "mitre": mitre}
        case _:
            raise ValueError("Dataname is incorrect")


def counts(data):
    match DATA_NAME:
        case "nvd" | "mitre":
            counts = metric_counts.calculate_metric_counts(
                data[DATA_NAME], CVSS_VERSION
            )
            return counts
        case "overlap" | "combined":
            nvd_counts = metric_counts.calculate_metric_counts(
                data["nvd"], CVSS_VERSION
            )
            mitre_counts = metric_counts.calculate_metric_counts(
                data["mitre"], CVSS_VERSION
            )
            return nvd_counts, mitre_counts
        case _:
            raise ValueError("Dataname is incorrect")


def main():
    parser = argparse.ArgumentParser(
        description="Bayes modelling of CVE to CVSS scores"
    )

    parser.add_argument(
        "--source",
        choices=DATA_TYPES,
        required=False,
        help="The source to use (mitre or nvd or combined or overlap)",
    )
    parser.add_argument(
        "--version",
        choices=[2, 30, 31],
        type=int,
        required=False,
        help="The CVSS version",
    )

    # Parse the arguments
    args = parser.parse_args()
    global DATA_NAME, CVSS_VERSION
    CVSS_VERSION = 31
    DATA_NAME = "combined"
    if args.source:
        DATA_NAME = args.source
    if args.version:
        CVSS_VERSION = args.version

    print(f"Using source: {args.source}, version: {args.version}")

    # data = {}
    # data["all"] = []
    # if data_name == "all":
    #     for data_set in DATA_TYPES:
    #         data_name = data_set
    #         if data_set == "all":
    #             continue
    #         data[data_name] = compute_gibbs()

    #     # data = temp_data.data()
    #     for metric_idx in range(len(data[DATA_TYPES[0]])):
    #         row_labels = data["mitre"][metric_idx]["columns"]
    #         column_labels = []
    #         for source in data.keys():
    #             if source == "all":
    #                 continue
    #             for item in data[source][metric_idx]["columns"]:
    #                 column_labels.append(item)

    #         data["all"].append(
    #             {
    #                 "columns": column_labels,
    #                 "caption": data[DATA_TYPES[0]][metric_idx]["caption"],
    #                 "label": data[DATA_TYPES[0]][metric_idx]["label"],
    #                 "row_labels": row_labels,
    #                 "data": np.hstack(
    #                     (
    #                         data["nvd"][metric_idx]["data"],
    #                         data["mitre"][metric_idx]["data"],
    #                         data["combined"][metric_idx]["data"],
    #                         data["overlap"][metric_idx]["data"],
    #                     )
    #                 ),
    #             }
    #         )

    #     with open("./table_data", "w") as file:
    #         file.write(str((data["all"])))
    #     for metric_idx in range(len(data["all"])):
    #         confusion_generator.generate(data["all"][metric_idx])

    # else:
    nvd_axes, mitre_axes = compute_metropolis_hastings()
    # nvd_figure = plt.figure(1)
    # mitre_figure = plt.figure(2)

    # for ax in nvd_axes:
    #     nvd_figure.add_subplot(ax)

    # for ax in mitre_axes:
    #     mitre_figure.add_subplot(ax)

    # nvd_figure.show()
    # mitre_figure.show()
    # for trace in results:
    #     print(trace)

    # summary = az.summary(trace)
    # print(summary)
    # for info in infos:
    #     confusion_generator.generate(info)


if __name__ == "__main__":
    main()
