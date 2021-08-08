"""
This script is part of the supporting information to the manuscript entitled
"Studying and mitigating the effects of data drifts on ML model performance at the example of chemical toxicity data"
Andrea Morger, Marina Garcia de Lomana, Ulf Norinder, Fredrik Svensson, Johannes Kirchmair, Miriam Mathea, and Andrea
Volkamer
It was last updated in August 2021.

# Helper functions
"""
import pandas as pd
import numpy as np
import sys
import math
import random
import copy
import os

import logging

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import StratifiedShuffleSplit

from nonconformist.nc import InverseProbabilityErrFunc, NcFactory

import umap

from continuous_calibration import (
    StratifiedRatioSampler,
    BalancedStratifiedRatioSampler,
    CrossValidationSampler,
    InductiveConformalPredictor,
    ContinuousCalibrationAggregatedConformalPredictor,
    CrossValidator,
)

logger = logging.getLogger(__name__)


# -----------------------
# Define helper functions
# -----------------------

# Load data


def load_data_per_endpoint(data_path, endpoint):
    """

    Parameters
    ----------
    data_path
    endpoint

    Returns
    -------

    """
    data = pd.read_csv(data_path, header=0, index_col=0)
    data = data[data[endpoint] != -1]

    y = data[endpoint].values
    columns = [
        col
        for col in data.columns
        if (not col.startswith("p0-"))
        and (not col.startswith("p1-"))
        and (not col.startswith("Toxicity"))
        and (col != endpoint)
        and (col != "SMILES (Canonical)")
        and (col != "smiles")
        and (col != "year")
        and (col != "REPORT_DATE")
        and (col != "PSN")
    ]
    print(len(columns))
    X = data[columns].values
    return X, y


def load_data_per_endpoint_incl_year(
    data_path, endpoint, descriptors, year=True, smiles=False
):
    """

    Parameters
    ----------
    smiles : bool
        True if smiles should be returned per compound
    year : bool
        True if a year column is present in the dataframe. Then, rows without a year (nan) will be dropped and
        an array with the years will be returned.
        Set to False if no 'year' column is present.

    data_path
    endpoint
    descriptors

    Returns
    -------

    """
    data = pd.read_csv(data_path, header=0, index_col=None)

    data = data[data[endpoint] != -1]
    # Some measurements don't have a year annotation, so they are dropped
    if year is True:
        data = data.dropna(subset=["year"])
    # Get descriptors (X)
    if descriptors == "chem":
        columns = [
            col
            for col in data.columns
            if (not col.startswith("p0-"))
            and (not col.startswith("p1-"))
            and (not col.startswith("Toxicity"))
            and (col != endpoint)
            and (col != "SMILES (Canonical)")
            and (col != "smiles")
            and (col != "year")
            and (col != "REPORT_DATE")
            and (col != "PSN")
            and (col != "molecule_chembl_id")
        ]
        print(endpoint, len(columns))
        # col lengths for [chembl, mnt, liver, chembl220/4078, chembl5763, chembl206, chembl279, chembl230, chembl340,
        # chembl240, chembl2039, chembl222, chembl228, 203]
        assert len(columns) in [
            2171,
            2093,
            1993,

            1816,
            1703,
            1881,
            1911,
            1728,
            2086,
            2108,
            1568,
            1649,
            1654,
            1853,
        ]
    elif descriptors == "bio":
        columns = [
            col
            for col in data.columns
            if (col.startswith("p0-")) or (col.startswith("p1-"))
        ]
        # col lengths for [chembl, mnt, liver, chembl220/4078, chembl5763, chembl206/2039, chembl279, chembl230,
        # chembl340, chembl240, chembl222, chembl228]
        assert len(columns) in [748, 694, 707, 679, 681, 676, 660, 672, 697, 687, 689, 690, 675]
    elif descriptors == "chembio":
        columns = [
            col
            for col in data.columns
            if (not col.startswith("Toxicity"))
            and (col != endpoint)
            and (col != "SMILES (Canonical)")
            and (col != "smiles")
            and (col != "year")
            and (col != "REPORT_DATE")
            and (col != "PSN")
            and (col != "molecule_chembl_id")
        ]
        print(endpoint, len(columns))
        # col lengths for [chembl, mnt, liver]
        assert len(columns) in [2919, 2787, 2700, 2495, 2384, 2557, 2571, 2400, 2783, 2795, 2244, 2338, 2344, 2528, 2788, 2701]
    else:
        columns = None
        logger.error("not implemented yet")

    print(len(columns))

    X = data[columns].values
    y = data[endpoint].values
    if year is False and smiles is False:
        return X, y
    elif year is True and smiles is False:
        year = data["year"].values
        return X, y, year
    elif year is False and smiles is True:
        smiles = data["smiles"].values
        return X, y, smiles
    elif year is True and smiles is True:
        year = data["year"].values
        smiles = data["smiles"].values
        return X, y, year, smiles


def filter_column_by_date(column_value, threshold):
    """

    Parameters
    ----------
    column_value
    threshold

    Returns
    -------

    """
    column_value = int(column_value[:4])
    if column_value <= threshold:
        return "update"
    elif column_value > threshold:
        return "test"


def load_update_test_data_per_endpoint(data_path, endpoint, threshold):
    """

    Parameters
    ----------
    data_path
    endpoint
    threshold

    Returns
    -------

    """
    data = pd.read_csv(data_path, header=0, index_col=0)
    data.dropna(subset=["date"], inplace=True)
    data["set"] = data["date"].apply(lambda x: filter_column_by_date(x, threshold))
    data_update = data[data["set"] == "update"]
    data_test = data[data["set"] == "test"]

    y_update = data_update[endpoint].values
    y_test = data_test[endpoint].values

    columns = [
        col
        for col in data.columns
        if (not col.startswith("p0-"))
        and (not col.startswith("p1-"))
        and (not col.startswith("Toxicity"))
        and (col != endpoint)
        and (col != "SMILES (Canonical)")
        and (col != "smiles")
        and (col != "date")
        and (col != "set")
    ]
    print(len(columns))
    X_update = data_update[columns].values
    X_test = data_test[columns].values
    return X_update, X_test, y_update, y_test


def load_scaffold_split(descriptor_file, scaffold_split_file):
    descriptor_df = pd.read_csv(descriptor_file)
    descriptor_df.dropna(subset=["year"], inplace=True)
    scaffold_split_df = pd.read_csv(scaffold_split_file)

    scaffold_split_dict = pd.Series(
        scaffold_split_df.Set.values, index=scaffold_split_df.molecule_chembl_id
    ).to_dict()
    descriptor_df["set"] = descriptor_df["molecule_chembl_id"].map(scaffold_split_dict)
    update1_mask = descriptor_df["set"] == "update1"
    update2_mask = descriptor_df["set"] == "update2"
    test_mask = descriptor_df["set"] == "test"

    return update1_mask, update2_mask, test_mask


# ML/CP


def prepare_rf_acp(
    normaliser_model, ntrees, n_folds_acp, random_state=None, smoothing=False
):
    """
    Prepare an acp with a random forest for continuous calibration

    Parameters
    ----------
    normaliser_model
    ntrees
    n_folds_acp
    random_state
    smoothing

    Returns
    -------

    """
    clf = RandomForestClassifier(n_estimators=ntrees, random_state=random_state)
    error_function = InverseProbabilityErrFunc()
    nc = NcFactory.create_nc(
        clf, err_func=error_function, normalizer_model=normaliser_model
    )
    icp = InductiveConformalPredictor(
        nc_function=nc,
        condition=(lambda instance: instance[1]),
        smoothing=smoothing,
        random_state=random_state,
    )
    if normaliser_model:
        ratio_sampler = BalancedStratifiedRatioSampler(
            n_folds=n_folds_acp, random_state=random_state
        )
    else:
        ratio_sampler = StratifiedRatioSampler(
            n_folds=n_folds_acp, random_state=random_state
        )
    acp = ContinuousCalibrationAggregatedConformalPredictor(
        predictor=icp, sampler=ratio_sampler, aggregation_func=np.median
    )
    return acp


def get_X_y_from_df(chembl_df):
    """
    get X (morgan descriptors) and y (labels) from a dataframe
    Parameters
    ----------
    chembl_df

    Returns
    -------

    """
    X = np.array(chembl_df.morgan.tolist())
    y = chembl_df.binary_value.values
    return X, y


# Evaluation


def calculate_deviation_square(error, sl):
    """
    Calculate square deviation, e.g. between (observed) error and significance level

    Parameters
    ----------
    error :
        observed error rate
    sl :
        significance level

    Returns
    -------
    square deviation
    """
    return (error - sl) ** 2


def calculate_rmsd_from_df(eval_df, cl=None):
    """

    Parameters
    ----------
    eval_df :
        evaluation df containing a column with the mean error rate
    cl :
        class

    Returns
    -------
    rmsd
    """
    if cl:
        eval_df["square"] = eval_df.apply(
            lambda row: calculate_deviation_square(
                row[f"error_rate_{cl} mean"], row["significance_level"]
            ),
            axis=1,
        )
    else:
        eval_df["square"] = eval_df.apply(
            lambda row: calculate_deviation_square(
                row["error_rate mean"], row["significance_level"]
            ),
            axis=1,
        )
    rmsd = np.round(math.sqrt(np.mean(eval_df["square"])), 3)
    return rmsd


def draw_calibration_plot_3_endpoints(
    endpoints,
    strategy,
    eval_dfs,
    colours=("darkolivegreen", "darkmagenta", "darkkhaki", "plum"),
    class_wise=True,
    efficiency=True,
    title_name=None,
):
    """
    Generate a calibration plot for exactly three endpoints, i.e. with exactly three subplots

    Parameters
    ----------
    endpoints
    strategy
    eval_dfs
    colours
    class_wise
    efficiency
    title_name

    Returns
    -------

    """

    plt.clf()
    fig, axs = plt.subplots(ncols=3, nrows=1)
    fig.set_figheight(7)
    fig.set_figwidth(20)

    yax = 0

    if class_wise and efficiency:
        evaluation_measures = [
            "error_rate_0",
            "error_rate_1",
            "efficiency_0",
            "efficiency_1",
        ]

    else:
        evaluation_measures = ["error_rate"]

    eval_legend = evaluation_measures.copy()

    for i, endpoint in enumerate(endpoints):
        eval_df = eval_dfs[i]
        axs[yax].plot([0, 1], [0, 1], "--", linewidth=1, color="black")
        sl = eval_df["significance_level"]
        for ev, colour in zip(evaluation_measures, colours):
            ev_mean = eval_df[f"{ev} mean"]
            ev_std = eval_df[f"{ev} std"]
            axs[yax].plot(sl, ev_mean, label=True, c=colour)
            axs[yax].fill_between(
                sl, ev_mean - ev_std, ev_mean + ev_std, alpha=0.3, color=colour
            )

        major_ticks = np.arange(0, 101, 20)
        minor_ticks = np.arange(0, 101, 5)

        axs[yax].set_xticks(minor_ticks / 100.0, minor=True)
        axs[yax].set_yticks(major_ticks / 100.0)
        axs[yax].set_yticks(minor_ticks / 100.0, minor=True)

        axs[yax].grid(which="minor", linewidth=0.5)  # alpha=0.5)
        axs[yax].grid(which="major", linewidth=1.5)  # alpha=0.9, linewidth=2.0)

        axs[yax].set_title(endpoint, fontsize=16)
        axs[yax].set_xlabel("significance")
        axs[yax].set_ylabel("error rate")
        par1 = axs[yax].twinx()
        par1.set_ylabel("efficiency (SCP)")

        yax += 1

    eval_legend.insert(0, "expected_error_rate")

    lgd = fig.legend(
        eval_legend, loc="lower center", bbox_to_anchor=(0.5, 1)
    )  # , 0.47))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(strategy, fontsize=20)

    return plt, lgd


def calc_n_cols_n_rows(n):
    """
    Calculate how many rows and columns are required in a subplot-plot, depending on the total number of subplots
    This function is used for several plots with subplots, e.g. calibration plots, line plots, umap plots etc.

    Parameters
    ----------
    n :
        number of subplots

    Returns
    -------

    """
    if n == 1:
        n_cols = 1
        n_rows = 1
    elif n == 2 or n == 4:
        n_cols = 2
        n_rows = n / 2
    else:
        n_cols = 3
        n_rows = math.ceil(n / 3)
    return int(n_cols), int(n_rows)


def draw_calibration_plot_more_endpoints(
    endpoints,
    strat,
    eval_dfs,
    colours=("blue", "darkred", "deepskyblue", "lightcoral"),
    class_wise=True,
    efficiency=True,
    title_name=None,
):
    """
    Create calibration plots for multiple datsets

    Parameters
    ----------
    endpoints
    strat
    eval_dfs
    colours
    class_wise
    efficiency
    title_name

    Returns
    -------

    """

    n_cols, n_rows = calc_n_cols_n_rows(len(endpoints))

    plt.clf()
    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows)
    fig.set_figheight(15)
    fig.set_figwidth(20)

    xax = 0
    yax = 0

    if class_wise and efficiency:
        evaluation_measures = [
            "error_rate_0",
            "error_rate_1",
            "efficiency_0",
            "efficiency_1",
        ]
    else:
        evaluation_measures = ["error_rate"]

    eval_legend = evaluation_measures.copy()
    eval_legend.insert(0, "expected_error_rate")

    for i, endpoint in enumerate(endpoints):
        eval_df = eval_dfs[i]

        if n_rows > 1:
            axs[xax, yax].plot([0, 1], [0, 1], "--", linewidth=1, color="black")
            sl = eval_df["significance_level"]
            for ev, colour in zip(evaluation_measures, colours):

                ev_mean = eval_df[f"{ev} mean"]
                ev_std = eval_df[f"{ev} std"]
                axs[xax, yax].plot(sl, ev_mean, label=True, c=colour)
                axs[xax, yax].fill_between(
                    sl, ev_mean - ev_std, ev_mean + ev_std, alpha=0.3, color=colour
                )

            major_ticks = np.arange(0, 101, 20)
            minor_ticks = np.arange(0, 101, 5)

            axs[xax, yax].set_xticks(minor_ticks / 100.0, minor=True)
            axs[xax, yax].set_yticks(major_ticks / 100.0)
            axs[xax, yax].set_yticks(minor_ticks / 100.0, minor=True)

            axs[xax, yax].grid(which="minor", linewidth=0.5)
            axs[xax, yax].grid(
                which="major", linewidth=1.5
            )

            axs[xax, yax].set_title(endpoint, fontsize=16)
            axs[xax, yax].set_xlabel("significance")
            axs[xax, yax].set_ylabel("error rate")
            par1 = axs[xax, yax].twinx()
            par1.set_ylabel("efficiency (SCP)")

            xax += 1
            if xax == n_rows:
                xax = 0
                yax += 1
        else:

            axs[yax].plot([0, 1], [0, 1], "--", linewidth=1, color="black")
            sl = eval_df["significance_level"]
            for ev, colour in zip(evaluation_measures, colours):
                ev_mean = eval_df[f"{ev} mean"]
                ev_std = eval_df[f"{ev} std"]
                axs[yax].plot(sl, ev_mean, label=True, c=colour)
                axs[yax].fill_between(
                    sl, ev_mean - ev_std, ev_mean + ev_std, alpha=0.3, color=colour
                )

            major_ticks = np.arange(0, 101, 20)
            minor_ticks = np.arange(0, 101, 5)

            axs[yax].set_xticks(minor_ticks / 100.0, minor=True)
            axs[yax].set_yticks(major_ticks / 100.0)
            axs[yax].set_yticks(minor_ticks / 100.0, minor=True)

            axs[yax].grid(which="minor", linewidth=0.5)
            axs[yax].grid(which="major", linewidth=1.5)

            axs[yax].set_title(endpoint, fontsize=16)
            axs[yax].set_xlabel("significance")
            axs[yax].set_ylabel("error rate")
            par1 = axs[yax].twinx()
            par1.set_ylabel("efficiency (SCP)")

            yax += 1

    lgd = fig.legend(eval_legend, loc="center left", bbox_to_anchor=(1, 0.47))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(strat, fontsize=20)

    return plt, lgd


mini_size = 8
small_size = 10
medium_size = 10


def boxplot_val_eff_acc_quer(
        evaluation_dfs,
        measures,
        significance_level,
        map_labels=False, fig_height=8.5, fig_width=17):
    """
    Make same plot but subplots next to each other instead of underneath
    """

    measure_dict_sl = {}
    for exp, dfs in evaluation_dfs.items():

        measure_dict_sl[exp] = {}
        for measure in measures:
            measure_dict_sl[exp][measure] = np.array([])
        for ep_df in dfs:
            ep_df_sl = ep_df[ep_df["significance_level"] == significance_level]
            for measure in measures:
                measure_dict_sl[exp][measure] = np.append(
                    measure_dict_sl[exp][measure],
                    ep_df_sl[f"{measure} mean"].values)

    experiments = evaluation_dfs.keys()

    if map_labels:
        labels_map_dict = {"cv_original": "cv",
                           "original": "cal_original",  # pred_holdout\n
                           "update": "iii",
                           "update1": "cal_update1",  # pred_holdout\n
                           "update2": "cal_update2",  # pred_holdout\n
                           "update12": "cal_update1_and_2",  # pred_holdout\n
                           }
        labels = [labels_map_dict[k] for k in experiments]
    else:
        labels = experiments

    boxprops = dict(linewidth=2)
    whiskerprops = dict(linewidth=2)
    capprops = dict(linewidth=2)
    medianprops = dict(linewidth=2, color='darkred')
    flierprops = dict(linewidth=2, markeredgecolor='darkred', markeredgewidth=2)

    plt.clf()
    fig, axs = plt.subplots(ncols=3)

    fig.set_figheight(cm2inch(fig_height))
    fig.set_figwidth(cm2inch(fig_width))
    plt.rc('xtick', labelsize=small_size)
    plt.rc('ytick', labelsize=small_size)
    plt.rc('legend', fontsize=small_size)
    yax = 0

    measures_map_dict = {"validity_bal": "Balanced validity", "efficiency_bal": "Balanced efficiency",
                         "accuracy_bal": "Balanced accuracy",
                         "validity_0": "Validity\n inactive class", "efficiency_0": "Efficiency\n inactive class",
                         "accuracy_0": "Accuracy\n inactive class", "validity_1": "Validity\n active class",
                         "efficiency_1": "Efficiency\n active class", "accuracy_1": "Accuracy\n active class"}

    for measure in measures:
        axs[yax].hlines(0.8, 1, len(experiments), linestyle="dashed")
        axs[yax].boxplot(
            [measure_dict_sl[exp][measure] for exp in experiments], labels=labels,
            widths=0.75, boxprops=boxprops,
            whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops, flierprops=flierprops,
        )
        axs[yax].set_xticklabels(labels, rotation=30, fontsize=small_size, ha="right")  # )"vertical")
        axs[yax].set_ylim(0.0, 1.0)

        if measure in measures_map_dict.keys():
            measure_title = measures_map_dict[measure]
        else:
            measure_title = measure
        axs[yax].set_title(measure_title, fontsize=small_size)

        yax += 1

    plt.tight_layout(h_pad=1, w_pad=1)

    return plt


def boxplot_and_df_for_eval_measure(
        evaluation_dfs,
        measure,
        descriptors,
        significance_level,
        data_results_path,
        datasets,
        name_spec=None,
):
    print(significance_level)
    """
    Create boxplots for a selected evaluation measure, e.g. validity, efficiency, rmsd etc.

    Parameters
    ----------
    evaluation_dfs
    measure
    descriptors
    significance_level
    data_results_path
    datasets

    Returns
    -------

    """
    measure_dict_sl = {}
    for strategy, dfs in evaluation_dfs.items():
        measure_dict_sl[strategy] = np.array([])
        for ep_df in dfs:
            ep_df_sl = ep_df[ep_df["significance_level"] == significance_level]
            measure_dict_sl[strategy] = np.append(
                measure_dict_sl[strategy], ep_df_sl[f"{measure} mean"].values
            )

    plt.clf()
    strategies = evaluation_dfs.keys()
    # if measure != "efficiency":
    plt.hlines(0.5, 0, 4, linestyle="dashed")

    plt.boxplot(
        [measure_dict_sl[strategy] for strategy in strategies], labels=strategies
    )

    plt.xticks(rotation="vertical")
    plt.gca().set_ylim(0.0, 1.0)
    plt.title(f"{measure} over all endpoints, {descriptors} descriptors, chembl")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if name_spec:
        plt.savefig(
            f"{data_results_path}/{measure}_{descriptors}_sl0{str(significance_level)[-1]}_{name_spec}_chembl.png"
        )
    else:
        plt.savefig(
            f"{data_results_path}/{measure}_{descriptors}_sl0{str(significance_level)[-1]}_chembl.png"
        )

    measure_dict_sl["dataset"] = datasets

    measure_df = pd.DataFrame.from_dict(measure_dict_sl)
    measure_df.set_index("dataset")
    if name_spec:
        measure_df.to_csv(
            f"{data_results_path}/{measure}_df_{descriptors}_sl0{str(significance_level)[-1]}_{name_spec}_chembl.csv"
        )
    else:
        measure_df.to_csv(
            f"{data_results_path}/{measure}_df_{descriptors}_sl0{str(significance_level)[-1]}_chembl.csv"
        )


# -----------------------------------------------
# Define threshold for train/update/test datasets
# -----------------------------------------------


def smaller_starting_year(year, starting_year):
    """
    If the year is older than the defined "starting_year", i.e
    the year from when we start counting, set it to the "starting_year"
    This is just internal handling of the code and won't change the data
    """
    if year < starting_year:
        year = starting_year
    return year


def count_nof_actives_inactives(df, starting_year, name=None):
    """
    Count # of actives and inactives per year
    """

    if name:
        df_years = copy.copy(df[["year", f"{name}_bioactivity"]])
    else:
        df_years = copy.copy(df[["year", "binary_value"]])
    df_years["year"] = df_years["year"].apply(
        lambda y: smaller_starting_year(y, starting_year)
    )

    years = range(starting_year, 2021)
    actives = []
    inactives = []
    for year in years:

        if name:
            act_year_df = df_years[
                (df_years["year"] == year) & (df_years[f"{name}_bioactivity"] == 1)
            ]
            inact_year_df = df_years[
                (df_years["year"] == year) & (df_years[f"{name}_bioactivity"] == 0)
            ]
        else:
            act_year_df = df_years[
                (df_years["year"] == year) & (df_years["binary_value"] == 1)
            ]
            inact_year_df = df_years[
                (df_years["year"] == year) & (df_years["binary_value"] == 0)
            ]

        actives.append(len(act_year_df))
        inactives.append(len(inact_year_df))
    return years, actives, inactives


def count_actives_inactives_all_datasets(
    dataset_dict, orig_path, general_df_name, starting_year=2000
):
    """
    Create a dict with the years, number of actives and inactives per dataset. This dict can later be used
    to plot the number of actives and inactives available per year.

    Parameters
    ----------
    dataset_dict :
        dict with dataset names as key
    orig_path :
        Path where dataframes are saved
    general_df_name :
        end name how descriptor df was called, e.g. "chembio.csv"
    starting_year :
        year from which to start counting. This is only required to save computational cost. We don't need to start
        iterating from year 1900 or even earlier since no data was published/added to chembl by then. Since even in
        2000 only very few data points were available, 2000 was used as a `starting_year`, i.e. to start iterating
         over years

    Returns
    -------

    """
    count_all_data_dict = {
        "target": [],
        "#standardised": [],
        "actives": [],
        "inactives": [],
    }
    count_dict = copy.copy(dataset_dict)
    for k, v in dataset_dict.items():
        # Define target_chembl_id

        # Load compounds
        bioact_df = pd.read_csv(os.path.join(orig_path, f"{k}_{general_df_name}"))
        # Get info about number of compounds available per target
        count_all_data_dict["target"].append(k)
        count_all_data_dict["#standardised"].append(bioact_df.shape[0])
        count_all_data_dict["actives"].append(
            bioact_df[bioact_df[f"{k}_bioactivity"] == 1].shape[0]
        )
        count_all_data_dict["inactives"].append(
            bioact_df[bioact_df[f"{k}_bioactivity"] == 0].shape[0]
        )

        # Count # actives and # inactives per year
        years, actives, inactives = count_nof_actives_inactives(
            bioact_df, starting_year=starting_year, name=k
        )

        count_dict[k] = [years, actives, inactives]

    return count_dict


def get_time_split_years(
    years, actives, inactives, train_thresh=0, update1_thresh=0, update2_thresh=0.0,
):
    """
    Create dictionaries per CP dataset (i.e. train, update, test) with the years and
    number of actives and inactives per CP dataset

    Parameters
    ----------
    years
    actives
    inactives
    train_thresh
    update1_thresh
    update2_thresh

    Returns
    -------

    """
    # Calculate total number of compounds available in the dataset
    total_compounds = np.sum(actives) + np.sum(inactives)
    print(total_compounds)
    # Define threshold based on total number of compound (percentage (ratio) threshold divided by total nof compounds))
    if train_thresh:
        train_thresh = train_thresh * total_compounds
    update1_thresh = update1_thresh * total_compounds
    if update2_thresh:
        update2_thresh = update2_thresh * total_compounds

    # Define 'new' threshold list, which is specific per dataset
    thresholds_list = [update1_thresh]
    if train_thresh:
        thresholds_list.insert(0, train_thresh)
    if update2_thresh:
        thresholds_list.append(update2_thresh)
    # thresholds_list = [train_thresh, update1_thresh, update2_thresh]
    if train_thresh:
        train_dict = {"years": [], "count_act": 0, "count_inact": 0}
    else:
        train_dict = None
    update1_dict = {"years": [], "count_act": 0, "count_inact": 0}
    if update2_thresh:
        update2_dict = {"years": [], "count_act": 0, "count_inact": 0}
    else:
        update2_dict = None
    test_dict = {"years": [], "count_act": 0, "count_inact": 0}

    collection_dict_list = [update1_dict, test_dict]
    if train_dict:
        collection_dict_list.insert(0, train_dict)
    if update2_dict:
        collection_dict_list.insert(2, update2_dict)
    collection_dict_index = 0

    for year, active, inactive in zip(years, actives, inactives):
        collection_dict = collection_dict_list[collection_dict_index]
        collection_dict["years"].append(year)
        collection_dict["count_act"] += active
        collection_dict["count_inact"] += inactive
        # If minimum threshold per dataset is reached (for actives and inactives):
        if collection_dict_index != (
            len(collection_dict_list) - 1
        ):  # If we are already at the test set,
            # don't increment, but collect rest of data here.

            if (
                collection_dict["count_act"] + collection_dict["count_inact"]
            ) >= thresholds_list[collection_dict_index]:
                collection_dict_index += 1
                # Increment index, so data are counted for next dataset

    if update2_dict:
        return train_dict, update1_dict, update2_dict, test_dict
    elif train_dict:
        return train_dict, update1_dict, test_dict
    else:
        return update1_dict, test_dict


def get_time_split_dict(
    count_dict,
    train_threshold=None,
    update1_threshold=None,
    update2_threshold=None,
    test_threshold=None,
    id="CHEMBLID",
):
    """

    Parameters
    ----------
    count_dict
    train_threshold
    update1_threshold
    update2_threshold
    test_threshold
    id

    Returns
    -------

    """
    datasets = ["update1", "test"]
    if train_threshold:
        datasets.insert(0, "train")
    if update2_threshold:
        datasets.insert(2, "update2")
        print(datasets)

    time_split_dict = {}

    for k, v in count_dict.items():
        print(k)
        time_split_dict[k] = []
        years, actives, inactives = count_dict[k]
        dataset_time_split_dicts = get_time_split_years(
            years,
            actives,
            inactives,
            train_threshold,
            update1_threshold,
            update2_threshold,
        )
        for dataset, ts_dict in zip(datasets, dataset_time_split_dicts):

            try:
                time_split_dict[k].append(ts_dict["years"][-1])
            except IndexError:
                time_split_dict[k].append(0)
        print(dataset_time_split_dicts[-1]["count_act"])
        print(dataset_time_split_dicts[-1]["count_inact"])
        assert dataset_time_split_dicts[-1]["count_act"] >= test_threshold
        assert dataset_time_split_dicts[-1]["count_inact"] >= test_threshold
    return time_split_dict


# ----------
# Line plots
# ----------


def cm2inch(cm):
    inch = 2.54
    return cm / inch


def draw_scatter_plot_one_endpoint(
        endpoint,
        evaluation_dfs_dict,
        evaluation_measures,
        colours=("navy", "green", "plum"),
        marker_styles=("o", "x", "+", "v"),
        significance_level=0.2,

):
    plt.clf()
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)
    plt.rc("legend", fontsize=12)

    measures_map_dict = {"validity_bal": "balanced validity",
                         "efficiency_bal": "balanced efficiency",
                         "accuracy_bal": "balanced accuracy",
                         }

    eval_legend = []
    for em in evaluation_measures:
        if em in measures_map_dict.keys():
            eval_legend.append(measures_map_dict[em])
        else:
            eval_legend.append(em)

        # Prepare data for plot, i.e. eval measure for each strategy
        measure_dict_sl = {"strategies": []}
        for meas in evaluation_measures:
            measure_dict_sl[meas] = []

        for strategy, df in evaluation_dfs_dict.items():
            measure_dict_sl["strategies"].append(strategy)
            df_sl = df[df["significance_level"] == significance_level]
            for meas in evaluation_measures:
                measure_dict_sl[meas].append(df_sl[f"{meas} mean"].values)

        strategies = measure_dict_sl["strategies"]

        labels_map_dict = {"cv_original": "cv",
                           "original": "cal_original",  # pred_holdout\n
                           "update1": "cal_update1",  # pred_holdout\n
                           "update2": "cal_update2",  # pred_holdout\n
                           }

        labels = [labels_map_dict[l] for l in strategies]

        for m, meas in enumerate(evaluation_measures):
            plt.scatter(
                strategies,
                measure_dict_sl[meas],
                color=colours[m],
                marker=marker_styles[m]
            )

        plt.hlines(0.8, 0, 2, linestyle="dashed")
        plt.xticks(np.arange(len(labels)), labels=labels, rotation=30, ha="right")
        plt.ylim(-0.05, 1.05)

        title = f"{endpoint}\n"
        plt.title(title, fontsize=14)

    lgd = plt.legend(eval_legend, loc='upper center', bbox_to_anchor=(0.5, 0.5), ncol=1, columnspacing=0.5,
                     numpoints=3)

    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])

    evaluation_measures = "_".join([em for em in evaluation_measures])

    return plt, lgd


def create_umap_data(
        chembl_id, descriptors_df, thresholds, dataset_colours, n_neighbors, min_distances, distance_metric,
        descriptors="chembio"
):

    descriptors_df = descriptors_df.dropna(subset=["year"])

    colours_numbers = descriptors_df["year"].apply(
        lambda x: colour_by_year(
            x, thresholds[0], thresholds[1], thresholds[2], colours=dataset_colours
        )
    )

    # Define columns used as descriptors (depending on the descriptor type, i.e. chem, bio or chembio)
    columns = columns_per_descriptor(
        descriptors_df.columns, descriptors, chembl_id
    )
    descriptors_df = descriptors_df[columns]
    all_descriptors = descriptors_df.values

    # Create embedding of UMAP fitted on all datapoints
    embedding = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_distances,
        metric=distance_metric,
        random_state=42  # Set random seed for reproducibility
    ).fit_transform(X=all_descriptors)

    return embedding, colours_numbers


def plot_umap(embedding, endpoint,
              colours_datasets,
              umap_colours,
              n_neighbor,
              min_distance,
              distance_metric,
              figsize=(17, 17), ):
    colours = colours_datasets
    print(len(colours))

    plt.figure(figsize=(cm2inch(figsize[0]), cm2inch(figsize[1])))
    plt.scatter(
        embedding[:, 0], embedding[:, 1], label=None, c=colours, s=10, alpha=0.5
    )
    plt.title(endpoint, fontsize=16)

    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    markers = [plt.Line2D([0, 0], [0, 0], color=colour, marker='o', linestyle='') for colour in umap_colours]
    plt.legend(markers, ["training set", "update1 set", "update2 set", "holdout set"], numpoints=1, loc="upper right")
    return plt


def draw_line_plot_more_datasets(
    evaluation_dfs_per_dataset,
    datasets,
    num_actives,
    num_inactives,
    descriptors,
    evaluation_measures,
    line_plots_path,
    strat,
    colours=("navy", "green", "plum"),
    figsize=(7.5, 5),
    significance_level=0.2,
):
    """
    Create a line plot in form of subplots for multiple datasets. Select a list of evaluation measures to be plotted in
    each subplot/per dataset

    Parameters
    ----------
    evaluation_dfs_per_dataset:
        A dict containing as keys the datasets and as values a dict
        containing as keys the strategy/experiment and as value the coresponding evaluation_df
    datasets: list
        dataset names
    num_actives: list
        number of actives (in training set) per dataset
    num_inactives: list
        number of inactives (in training set) per dataset
    descriptors:
        type of descriptors used for experiment, i.e chem, bio or chembio
    evaluation_measures:
        evaluation measures to be included in the plot
    line_plots_path:
        path to save plots
    strat:
        strategy/experiments to include in plot, i.e. "all", "no_norm" (only experiments without normalisation)
        or "norm" (only experiments with normalisation)
    colours:
        colours for lines
    figsize:
        size of complete plot
    significance_level:
        significance level, at which the evaluation should be performed/plotted

    Returns
    -------

    """
    n_cols, n_rows = calc_n_cols_n_rows(len(datasets))

    plt.clf()
    plt.rc("xtick", labelsize=7)
    plt.rc("ytick", labelsize=7)
    plt.rc("legend", fontsize=7)
    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows)
    fig.set_figheight(cm2inch(figsize[0]))
    fig.set_figwidth(cm2inch(figsize[1]))

    xax = 0
    yax = 0

    eval_legend = evaluation_measures.copy()

    for i, dataset in enumerate(datasets):

        dataset_evaluation_dfs = evaluation_dfs_per_dataset[dataset]

        # Prepare data for plot, i.e. eval measure for each strategy
        measure_dict_sl = {}
        measure_dict_sl["strategies"] = []
        for meas in evaluation_measures:
            measure_dict_sl[meas] = []

        for strategy, df in dataset_evaluation_dfs.items():
            measure_dict_sl["strategies"].append(strategy)
            df = df[0]
            df_sl = df[df["significance_level"] == significance_level]
            for meas in evaluation_measures:
                measure_dict_sl[meas].append(df_sl[f"{meas} mean"].values)

        strategies = measure_dict_sl["strategies"]

        if n_rows > 1:
            axs[xax, yax].hlines(0.5, 0, 9, linestyle="dashed")

            for m, meas in enumerate(evaluation_measures):
                axs[xax, yax].plot(
                    strategies,
                    measure_dict_sl[meas],
                    color=colours[m],
                    linewidth=0.5,
                    marker="o",
                )

            axs[xax, yax].set_xticklabels(strategies, rotation=90)
            axs[xax, yax].set_ylim(0.0, 1.0)
            title = f"{dataset}, {num_actives[i]} actives, {num_inactives[i]} inactives"
            axs[xax, yax].set_title(title, fontsize=7)

            xax += 1
            if xax == n_rows:
                xax = 0
                yax += 1
        else:
            axs[yax].hlines(0.5, 0, 9, linestyle="dashed")

            for m, meas in enumerate(evaluation_measures):
                axs[yax].plot(
                    strategies,
                    measure_dict_sl[meas],
                    color=colours[m],
                    linewidth=0.5,
                    marker="o",
                )

            axs[yax].set_xticklabels(strategies, rotation=90)
            axs[yax].set_ylim(0.0, 1.0)
            title = f"{dataset}, {num_actives[i]} actives, {num_inactives[i]} inactives"
            print("title", title)
            axs[yax].set_title(title, fontsize=10)

            yax += 1

    lgd = fig.legend(eval_legend, loc='upper center', bbox_to_anchor=(0.51, 0.05), ncol=5, columnspacing=0.9,
                     numpoints=3)

    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
    fig.suptitle(
        f"{descriptors} descriptors, {', '.join(evaluation_measures)}", fontsize=10
    )

    evaluation_measures = "_".join([em for em in evaluation_measures])
    plt.savefig(
        f"{line_plots_path}/line_plot_{descriptors}_{strat}_{evaluation_measures}.pdf", format="pdf",
        bbox_extra_artists=(lgd,),
                bbox_inches='tight')

    return plt, lgd


# ----
# UMAP
# ----


def colour_by_year(year, train_thresh, update1_thresh, update2_thresh, colours=None):
    """
    Assign/return a colour depending on the year the data point was published.

    Parameters
    ----------
    year :
        publication year of data point
    train_thresh :
        Last year threshold to assign to training set
    update1_thresh :
        Last year threshold to assign to update1 set
    update2_thresh :
        Last year threshold to assign to update2 set
    colours :
        List of colours for training, update1, update2 and test set

    Returns
    -------
    Colour based on the publication year
    """

    if colours is None:
        colours = ["navy", "plum", "mediumaquamarine", "green"]

    if year <= train_thresh:
        return colours[0]
    elif year <= update1_thresh:
        return colours[1]
    elif year <= update2_thresh:
        return colours[2]
    elif year <= 2020:
        return colours[3]


def columns_per_descriptor(input_columns, descriptors, chembl_id):
    """
    Define which columns to collect for descriptor, depending on selected descriptor type

    Parameters
    ----------
    input_columns :
        List of all columns to select from
    descriptors :
        Descriptor type
    chembl_id :
        chembl_id of the dataset (required because bioactivity is named after chembl_id)

    Returns
    -------

    """
    print("desc", descriptors)
    if descriptors == "chem":
        output_columns = [
            col
            for col in input_columns
            if (not col.startswith("p0-"))
            and (not col.startswith("p1-"))
            and (not col.startswith("Toxicity"))
            and (col != f"{chembl_id}_bioactivity")
            and (col != "SMILES (Canonical)")
            and (col != "smiles")
            and (col != "year")
            and (col != "molecule_chembl_id")
        ]
        print(len(output_columns))
        assert len(output_columns) == 2171
    elif descriptors == "bio":
        output_columns = [
            col
            for col in input_columns
            if (col.startswith("p0-")) or (col.startswith("p1-"))
        ]
        print(len(output_columns))
        assert len(output_columns) == 746
    elif descriptors == "chembio":
        output_columns = [
            col
            for col in input_columns
            if (not col.startswith("Toxicity"))
            and (col != f"{chembl_id}_bioactivity")
            and (col != "SMILES (Canonical)")
            and (col != "smiles")
            and (col != "year")
            and (col != "molecule_chembl_id")
        ]

    else:
        print(f"column selection for {descriptors} descriptors not implemented yet")
        output_columns = None

    return output_columns


def plot_umap_more_datasets(
    datasets,
    descriptors,
    colours_datasets,
    umap_save_path,
    n_neighbor,
    min_distance,
    distance_metric,
    figsize=(15, 20),
):
    """
    Draw a plot from the umap embeddings for all datasets, based on a list of colours for the individual datasets

    Parameters
    ----------
    datasets :
        list of datasets to include in plot as subplot
    descriptors :
        descriptor type used to create embedding, i.e. chem, bio or chembio
    colours_datasets :
        list of colours (based on dataset)
    umap_save_path :
        path to save umap
    figsize :
        size of overall figure

    Returns
    -------
    UMAP plot
    """
    n_cols, n_rows = calc_n_cols_n_rows(len(datasets))

    plt.clf()
    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows)

    fig.set_figheight(figsize[0])
    fig.set_figwidth(figsize[1])

    xax = 0
    yax = 0

    for i, dataset in enumerate(datasets):
        print(i)
        colours = colours_datasets[i]
        print(len(colours))

        embedding = np.load(
            f"{umap_save_path}umap_embedding_{descriptors}_{dataset}_{n_neighbor}_neighbours_0{str(min_distance).split('.')[1]}_distance_{distance_metric}.npy"
        )

        if n_rows > 1:
            axs[xax, yax].scatter(
                embedding[:, 0], embedding[:, 1], label=None, c=colours, s=10, alpha=0.5
            )
            axs[xax, yax].set_title(dataset, fontsize=16)

            xax += 1
            if xax == n_rows:
                xax = 0
                yax += 1
        else:
            axs[yax].scatter(
                embedding[:, 0],
                embedding[:, 1],
                label=None,
                c=colours_numbers,
                s=3,
                alpha=0.2,
            )
            axs[yax].set_title(f"{dataset}, {descriptors}", fontsize=16)

            yax += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(descriptors, fontsize=20)

    return plt
