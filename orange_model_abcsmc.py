import numpy as np
import pandas as pd
import pyabc
import tempfile
import os
from pyabc import ABCSMC, RV, Distribution
import matplotlib.pyplot as plt
from pyabc import History
from pyabc.visualization import plot_kde_1d, plot_kde_1d_highlevel
import scipy.stats as stats
import argparse


"""
Third version: 
lysis_combo parameter = lysis_1664_535 + delta_combo
"""


# parameters = {"entry_combo": x1, "entry_535_1664": x2, "rep_combo": x3,"delta_combo": x4 , "lysis_1664_535": x5,
#          "rei_combo_1664": x6, "rei_535": x7}
# "lysis_combo" = lysis_664_535 + delta_combo


def set_parameters_in_arr(parameters):  # dict of 7 params to be inferred, np array of all parameters
    parameters_df = pd.read_csv("initial_coef_no_1691.csv", header=[0], index_col=[0])

    parameters_df.loc["combo_1664", "entry"] = parameters["entry_combo"]
    parameters_df.loc["535_1664", "entry"] = parameters["entry_535"]
    parameters_df.loc["535", "entry"] = parameters["entry_535"]
    parameters_df.loc["combo_1664", "rep"] = parameters["replication_combo"]
    parameters_df.loc["535_1664", "lysis"] = parameters["lysis_delay_1664"]
    parameters_df.loc["combo_1664", "lysis"] = parameters["added_lysis_delay_combo"] + parameters["lysis_delay_1664"]
    parameters_df.loc["1664", "lysis"] = parameters["lysis_delay_1664"]
    parameters_df.loc["combo_1664", "reinfection"] = parameters["reinfection_1664"]
    parameters_df.loc["535_1664", "reinfection"] = parameters["reinfection_1664"]
    parameters_df.loc["535", "reinfection"] = parameters["reinfection_535"]
    parameters_df.loc["1664", "reinfection"] = parameters["reinfection_1664"]

    param_arr = parameters_df.to_numpy()
    return param_arr


def normalize(simulation):
    """
    normalize the values of frequencies returned by simulation
    """
    s = np.sum(simulation, axis=0)
    normalized_sim = simulation / s
    return normalized_sim


def calc_isolated_muts(normalized_simulation):
    """
    create a matrix (np.array) of the simulated data of each isolated mutation
    """
    m_535 = normalized_simulation[1, :] + normalized_simulation[2, :]  # 535_1664 + 535
    m_1664 = (
        normalized_simulation[0, :] + normalized_simulation[1, :] + normalized_simulation[3, :]
    )  # combo + 535_1664 + 1664
    m_combo = normalized_simulation[0, :]  # combo
    wt = normalized_simulation[4, :]  # wt
    isolated_muts = np.vstack((m_535, m_1664, m_combo, wt))
    return isolated_muts


def model(parameters):
    """
    parameters: dict of the current set of parameters {name_of_param: val_of_param}
    run the simulation:
    set the current set of parameters
    create each time point by multiplying the prev time point and the relevant parameters
    normalize simulation values into frequencies
    isolate mutations

    returns: np.array of the simulated isolated mutations in a dictionary
    """
    parameters_arr = set_parameters_in_arr(parameters)

    # calc simulation
    time_0 = np.array([0.342047, 0.1637, 0.05, 0.42325, 0.021003])  # (combo_1664, 535_1664, 535, 1664, WT)
    time_15 = time_0 * parameters_arr[:, 0] * parameters_arr[:, 1]  # entry * replication  --> replication was added
    time_30 = time_15 * parameters_arr[:, 0] * parameters_arr[:, 1]  # entry * replication
    time_45 = time_30 * parameters_arr[:, 1]  # replication
    time_60 = time_45 * parameters_arr[:, 1] * parameters_arr[:, 2]  # replication * lysis --> lysis was added
    time_75 = time_60 * parameters_arr[:, 2] * parameters_arr[:, 3]  # lysis * reinfection
    time_90 = time_75 * parameters_arr[:, 2] * parameters_arr[:, 3]  # lysis * reinfection

    simulation = np.vstack((time_0, time_15, time_30, time_45, time_60, time_90))  # time_75 is excluded
    simulation = np.transpose(simulation)
    normalized_simulation = normalize(simulation)
    isolated_muts = calc_isolated_muts(normalized_simulation)

    return {"isolated_muts": isolated_muts}


def distance(simulation, data):
    """
    distance function: squared diff
    """
    return np.sum((data["observed_data"] - simulation["isolated_muts"]) ** 2)


"""
parameters priors: 
all uniform(0,4) except for added_lysis_delay_combo which is U(-1,1)
"""


def weighted_percentile(data, percentiles, weights=None):
    if weights is None:
        return np.percentile(data, percentiles)

    sorter = np.argsort(data)
    data, weights = np.array(data)[sorter], np.array(weights)[sorter]
    cumsum = np.cumsum(weights)
    return np.interp(np.array(percentiles) / 100.0 * cumsum[-1], cumsum, data)


def plot_posteriors_with_priors(history, parameter_priors_dict, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each parameter
    for parameter, val in parameter_priors_dict.items():
        prior = val[1:]
        param_name = val[0]
        fig, ax = plt.subplots()

        # Plot prior as a filled area
        x = np.arange(prior[0], prior[1], 0.001)
        y = stats.uniform.pdf(x, loc=prior[0], scale=prior[1] - prior[0])
        ax.fill_between(x, y, color="gray", alpha=0.3, label="Prior")

        df, w = history.get_distribution()
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=prior[0],  # parameter_priors_dict[parameter]
            xmax=prior[1],
            x=parameter,
            ax=ax,
            label="posterior",
        )

        # Calculate 95% credible interval
        # credible_interval = np.percentile(df[parameter], [2.5, 97.5], axis=0)
        credible_interval = weighted_percentile(df[parameter], [2.5, 97.5], weights=w)

        # Calculate the median of the posterior distribution
        median = weighted_percentile(df[parameter], 50, weights=w)

        # Plot the credible interval
        ax.axvline(credible_interval[0], color="blue", linestyle="--", label="95% CI Lower")
        ax.axvline(credible_interval[1], color="green", linestyle="--", label="95% CI Upper")

        # Plot the median of the posterior distribution
        ax.axvline(median, color="red", linestyle="-", label="Median")

        # Add labels and legend
        ax.set_xlabel(param_name, fontsize=20)
        ax.set_ylabel("Density", fontsize=20)

        # Increase the size of the tick labels on x and y axes
        ax.tick_params(axis="both", which="major", labelsize=16)

        # Move the legend outside the plot
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2, fontsize=20)
        # ax.legend(fontsize="16")

        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{parameter}_posterior.png"))
        plt.close(fig)


def plot_all_posteriors_with_priors(history, parameter_priors_dict, output_dir):
    """
    Plot all posterior distributions with their priors in a single figure with subplots.

    :param history: History object containing posterior results
    :param parameter_priors_dict: Dictionary with prior distributions for each parameter
    :param output_dir: Directory to save the figure
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Determine the number of parameters
    n_params = len(parameter_priors_dict)
    n_cols = 2  # Number of columns in the grid
    n_rows = (n_params + 1) // n_cols  # Number of rows needed

    # Create a figure with subplots arranged in a grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 5))

    # Flatten axes in case of single row
    axes = axes.flatten()

    # Iterate over each parameter and its prior
    for i, (parameter, val) in enumerate(parameter_priors_dict.items()):
        prior = val[1:]  # Prior range
        param_name = val[0]  # Parameter name
        ax = axes[i]

        # Plot prior as a filled area
        x = np.arange(prior[0], prior[1], 0.001)
        y = stats.uniform.pdf(x, loc=prior[0], scale=prior[1] - prior[0])
        ax.fill_between(x, y, color="gray", alpha=0.3, label="Prior")

        # Get the posterior distribution
        df, w = history.get_distribution()

        # Plot posterior distribution using kernel density estimation (KDE)
        pyabc.visualization.plot_kde_1d(
            df,
            w,
            xmin=prior[0],
            xmax=prior[1],
            x=parameter,
            ax=ax,
            label="Posterior",
        )

        # Calculate the credible interval and median
        credible_interval = weighted_percentile(df[parameter], [2.5, 97.5], weights=w)
        median = weighted_percentile(df[parameter], 50, weights=w)

        # Plot credible interval and median
        ax.axvline(credible_interval[0], color="blue", linestyle="--", label="95% CI Lower")
        ax.axvline(credible_interval[1], color="green", linestyle="--", label="95% CI Upper")
        ax.axvline(median, color="red", linestyle="-", label="Median")

        # Set axis labels
        ax.set_xlabel(param_name, fontsize=22)
        ax.set_ylabel("Density", fontsize=20)

        # Increase the size of tick labels on x and y axes
        ax.tick_params(axis="both", which="major", labelsize=18)

    # Hide any unused axes (if n_params is not a perfect square)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Move the shared legend outside the plot
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=3,
        fontsize=22,
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the legend
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    # Save the figure
    plt.savefig(os.path.join(output_dir, "all_posteriors_with_priors.png"), bbox_inches="tight")
    plt.show()


def plot_posterior_vs_observed(history: History, observed_data: np.ndarray, output_dir, n_samples: int = 100):
    """
    Plot simulated data from posterior distributions against observed data.

    :param history: pyABC History object containing the ABC-SMC results
    :param observed_data: Numpy array of observed data
    :param n_samples: Number of posterior samples to use for simulation
    """
    # Get the final population
    df, w = history.get_distribution(m=0, t=history.max_t)

    # Sample from the posterior
    samples = df.sample(n=n_samples, weights=w, replace=True)

    # Run simulations
    simulations = []
    for _, sample in samples.iterrows():
        sim = model(sample)["isolated_muts"]
        simulations.append(sim)

    # Calculate mean and credible intervals
    sim_mean = np.mean(simulations, axis=0)
    sim_low = np.percentile(simulations, 2.5, axis=0)
    sim_high = np.percentile(simulations, 97.5, axis=0)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(25, 15), sharex=True)
    mutation_names = ["A535G turquoise", "A1664G orange", "A1664G + purple combo", "WT"]
    # time_points = ["0", "15", "30", "45", "60", "90"]  # 75 need to be added??

    time_points = [0, 15, 30, 45, 60, 90]

    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes, mutation_names)):
        ax.plot(time_points, observed_data[i], "ro-", label="Observed")  # range(6)
        ax.plot(time_points, sim_mean[i], "b-", label="Simulated (mean)")  # range(6)
        ax.fill_between(time_points, sim_low[i], sim_high[i], color="b", alpha=0.2, label="95% CI")  # range(6)
        ax.set_ylabel(f"{name} freq.", fontsize=26)
        ax.set_xticks(time_points)  # range(6)
        ax.set_xticklabels([str(tick) for tick in time_points])  # time_points
        # ax.legend(fontsize="22")

        ax.tick_params(axis="both", which="major", labelsize=22)

    for ax in axes[2:]:
        ax.set_xlabel("Time (minutes)", fontsize=26)
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    # Create a shared legend outside the subplots
    handles, labels = axes[0].get_legend_handles_labels()  # Get the handles and labels from one of the axes
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=1,
        fontsize=24,
    )

    plt.savefig(os.path.join(output_dir, "fig_sim_vs_emp.png"), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str)
    parser.add_argument("population_size", type=int, default=10000)
    parser.add_argument("minimum_epsilon", type=float, default=0.05)
    parser.add_argument("max_nr_populations", type=int, default=15)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    arguments_filename = os.path.join(args.output_dir, "arguments.md")
    with open(arguments_filename, "w") as f:
        f.write(
            f"population_size={args.population_size}\n"
            f"minimum_epsilon={args.minimum_epsilon}\n"
            f"max_nr_populations={args.max_nr_populations}"
        )

    # priors:
    parameter_priors = Distribution(
        entry_combo=RV("uniform", 0, 4),
        entry_535=RV("uniform", 0, 4),
        replication_combo=RV("uniform", 0, 4),
        added_lysis_delay_combo=RV("uniform", -1, 2),
        lysis_delay_1664=RV("uniform", 0, 4),
        reinfection_1664=RV("uniform", 0, 4),
        reinfection_535=RV("uniform", 0, 4),
    )

    observed_data_df = pd.read_csv("new_empirical_no_1691.csv", header=[0], index_col=[0])
    observed_data = observed_data_df.to_numpy()

    db_path = "sqlite:///" + os.path.join(tempfile.gettempdir(), "test.db")

    # Set up the ABCSMC object
    abc = pyabc.ABCSMC(
        models=model,
        parameter_priors=parameter_priors,
        distance_function=distance,
        population_size=args.population_size,
    )

    # Initialize the ABCSMC run
    abc.new(db_path, {"observed_data": observed_data})

    # Run the inference
    history = abc.run(minimum_epsilon=args.minimum_epsilon, max_nr_populations=args.max_nr_populations)

    plot_posterior_vs_observed(history, observed_data, args.output_dir)

    parameter_wider_priors_dict = {
        "entry_combo": [r"$\omega_{e,oc}$", 0, 4],
        "entry_535": [r"$\omega_{e,tu}$", 0, 4],
        "replication_combo": [r"$\omega_{r,oc}$", 0, 4],
        "added_lysis_delay_combo": [r"$\delta$", -1, 1],
        "lysis_delay_1664": [r"$\omega_{l,o}$", 0, 4],
        "reinfection_1664": [r"$\omega_{rei,o}$", 0, 4],
        "reinfection_535": [r"$\omega_{rei,tu}$", 0, 4],
    }

    plot_all_posteriors_with_priors(history, parameter_wider_priors_dict, args.output_dir)

    # get epsilons:
    epsilons = history.get_all_populations()["epsilon"]

    # Save to a CSV file
    output_file_path = os.path.join(args.output_dir, "epsilons.csv")
    epsilons.to_csv(output_file_path, index=False)

    posterior_samples, w = history.get_distribution(m=0, t=history.max_t)

    # Access the parameter samples (stored in a pandas DataFrame)
    parameter_samples = posterior_samples

    # Define your threshold for each parameter
    threshold = 1

    # Check if each parameter sample is above or below the threshold
    # Let's assume you're checking this for one specific parameter, e.g., "param_1"
    parameter_entry_combo = "entry_combo"
    parameter_rep_combo = "replication_combo"
    above_threshold_entry = parameter_samples[parameter_entry_combo] > threshold
    below_threshold_entry = parameter_samples[parameter_entry_combo] <= threshold

    above_threshold_rep = parameter_samples[parameter_rep_combo] > threshold
    below_threshold_rep = parameter_samples[parameter_rep_combo] <= threshold

    # Calculate the proportion of samples above the threshold
    proportion_above_entry = np.mean(above_threshold_entry)
    proportion_below_entry = np.mean(below_threshold_entry)

    proportion_above_rep = np.mean(above_threshold_rep)
    proportion_below_rep = np.mean(below_threshold_rep)

    # Define your output file
    output_file = os.path.join(args.output_dir, "posterior_results.txt")

    # Open the file in write mode
    with open(output_file, "w") as f:
        # Write the results to the file
        f.write(f"Proportion of samples for {parameter_entry_combo} above {threshold}: {proportion_above_entry}\n")
        f.write(
            f"Proportion of samples for {parameter_entry_combo} below or equal to {threshold}: {proportion_below_entry}\n"
        )

        f.write(f"Proportion of samples for {parameter_rep_combo} above {threshold}: {proportion_above_rep}\n")
        f.write(
            f"Proportion of samples for {parameter_rep_combo} below or equal to {threshold}: {proportion_below_rep}\n"
        )


## dir: orange_no_1691. generations: 15, particles: 10000, epsilon: 0.1
## dir: orange_no_1691_new, generations: 15, particles: 10000, epsilon: 0.05
## dir: orange_no_1691_new2, generations: 15, particles: 10000, epsilon: 0.05
## dir: orange_no_1691_new75, generations: 15, particles: 10000, epsilon: 0.05
