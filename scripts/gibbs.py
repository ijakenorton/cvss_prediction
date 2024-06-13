import numpy as np
import pymc3 as pm
import theano.tensor as tt
import argparse
from cvss_types import ConfusionMatrixInfo, Metrics
import confusion_generator
import utils
import overlap

# from scipy.stats import dirichlet
import numpy as np
import metric_counts

import seaborn as sns
import arviz as az
import matplotlib.pyplot as plt

DATA_TYPES = ["mitre", "nvd", "combined", "overlap", "all"]
data_name = ""
CVSS_VERSION = "3.1"

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


def plot_confusion_matrix(
    trace, var_name, categories, title="Confusion Matrix", outpath="default"
):

    cm_mean = np.mean(trace.posterior[var_name].values, axis=(0, 1))

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
    plt.savefig(outpath)
    # plt.show()


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

    nvd_observed = {
        metric: np.array(list(data.values())) for metric, data in nvd.items()
    }

    mitre_observed = {
        metric: np.array(list(data.values())) for metric, data in mitre.items()
    }

    results = {}
    for metric in nvd_observed:
        nvd_counts = nvd_observed[metric]
        mitre_counts = mitre_observed[metric]
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
            categories = metrics_and_dimensions[metric]

            plot_confusion_matrix(
                trace,
                "pi_mitre",
                categories,
                title=f"Confusion Matrix for {metric} from Mitre version {CVSS_VERSION}",
                outpath=f"mitre_{metric}_{CVSS_VERSION}.png",
            )

            # Plot confusion matrix for NVD
            plot_confusion_matrix(
                trace,
                "pi_nvd",
                categories,
                title=f"Confusion Matrix for {metric} from NVD version {CVSS_VERSION}",
                outpath=f"nvd_{metric}_{CVSS_VERSION}.png",
            )

            summary = az.summary(trace)
            print(summary)
            # print(summary)
            #         results[metric] = trace

    # with open(f"../results/{data_name}_metropolis", "w") as file:
    #     for metric, result in results.items():
    #         file.write(f"Results for {metric}:\n")
    #         file.write(f"Probabilities: result['probabilities']\n")
    #         file.write(f"Confusion Matrix:\n {result['confusion_matrix']}\n")
    #         file.write("\n")


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
    with open(f"../results/{data_name}_gibbs_formatted", "w") as file:
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
                "label": f"table:{data_name}-{metric}",
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
    with open(f"../results/{data_name}_bayes", "w") as file:
        for i, (prior, observed) in enumerate(zip(confusion_matrices, observed_totals)):
            updated_prior = prior.diagonal() + observed
            posterior_samples = dirichlet(updated_prior).rvs(size=1000)
            expected_probabilities = posterior_samples.mean(axis=0)
            file.write(
                f"Expected Probabilities for '{list(metrics_and_dimensions.keys())[i]}':{expected_probabilities}\n"
            )
            file.write("\n")


def read_data():
    match data_name:
        case "nvd" | "mitre":
            data = utils.read_data(f"../data/{data_name}_2.0_cleaned.pkl")
            if data is None:
                raise ValueError(f"Data not found for {data_name}")
            return {data_name: data["data"]}
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
    match data_name:
        case "nvd" | "mitre":
            counts: Metrics = metric_counts.calculate_metric_counts(data[data_name])
            return counts
        case "overlap" | "combined":
            nvd_counts: Metrics = metric_counts.calculate_metric_counts(data["nvd"])
            mitre_counts: Metrics = metric_counts.calculate_metric_counts(data["mitre"])
            # for metric in nvd_counts:
            #     for answer in nvd_counts[metric]:
            #         nvd_counts[metric][answer] += mitre_counts[metric][answer]
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
        required=True,
        help="The source to use (mitre or nvd or combined or overlap)",
    )

    # Parse the arguments
    args = parser.parse_args()
    global data_name
    data_name = args.source
    print(f"Using source: {args.source}")

    data = {}
    data["all"] = []
    if data_name == "all":
        for data_set in DATA_TYPES:
            data_name = data_set
            if data_set == "all":
                continue
            data[data_name] = compute_gibbs()

        # data = temp_data.data()
        for metric_idx in range(len(data[DATA_TYPES[0]])):
            row_labels = data["mitre"][metric_idx]["columns"]
            column_labels = []
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
        results = compute_metropolis_hastings()
        # for trace in results:
        #     print(trace)

        # summary = az.summary(trace)
        # print(summary)
        # for info in infos:
        #     confusion_generator.generate(info)


if __name__ == "__main__":
    main()
