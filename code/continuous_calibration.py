"""
Conformal Prediction

Parts of this code are taken from https://github.com/donlnz/nonconformist/blob/master/nonconformist/acp.py
and/or adapted for the use of this project
"""
import pandas as pd
import numpy as np

import random
import copy

# import logging
# import json
# import sys
# from collections import OrderedDict
# from types import GeneratorType
#
# import sklearn
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsRegressor, _kd_tree
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

# from sklearn.base import clone
# from sklearn.tree._classes import DecisionTreeClassifier
# from nonconformist.base import ClassifierAdapter, RegressorAdapter
from nonconformist.icp import IcpClassifier

# from nonconformist.nc import (
#     InverseProbabilityErrFunc,
#     MarginErrFunc,
#     NcFactory,
#     ClassifierNc,
#     RegressorNormalizer,
# )
import matplotlib.pyplot as plt

# --------------------------------
# Samplers
# --------------------------------


class Sampler:
    """
    Basic 'sampler' class, to generate samples/subsets for the different conformal prediction steps
    """

    def _gen_samples(self, y):
        raise NotImplementedError("Implement in your subclass")
        pass

    def gen_samples(self, labels):
        """

        Parameters
        ----------
        labels : pd.Series
            a series of labels for the molecules

        Returns
        -------

        """
        y = labels
        return self._gen_samples(y)  # values are in form of array

    def _balance(self, y_idx, idx, ratio=1.0):
        # Mask to distinguish compounds of inactive and active class of dataset
        mask_0 = y_idx == 0
        y_0 = idx[mask_0]
        mask_1 = y_idx == 1
        y_1 = idx[mask_1]

        # Define which class corresponds to larger proper training set and is subject to undersampling
        larger = y_0 if y_0.size > y_1.size else y_1
        smaller = y_1 if y_0.size > y_1.size else y_0
        # Subsample larger class until same number of instances as for smaller class is reached
        np.random.seed(self.random_state)
        while smaller.size < larger.size / ratio:
            k = np.random.choice(range(larger.size))
            larger = np.delete(larger, k)

            idx = sorted(np.append(larger, smaller))
        assert len(idx) == 2 * len(smaller)

        return idx

    @property
    def name(self):
        raise NotImplementedError("Implement in your subclass")


class StratifiedRatioSampler(Sampler):
    """
    This sampler can e.g. be used for aggregated conformal predictors

    Parameters
    ----------
    test_ratio : float
        Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split
        Note: according to sklearn, test_ratio could also be int or None. To allow this, we would need to remove
              the assert statement
    n_folds : int
        Number of re-shuffling and splitting iterations.

    Attributes
    ----------
    test_ratio : float
        Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split
        Note: according to sklearn, test_ratio could also be int or None. To allow this, we would need to remove
              the assert statement
    n_folds : int
        Number of re-shuffling and splitting iterations.

    Examples
    --------
    todo
    """

    def __init__(self, test_ratio=0.3, n_folds=1, random_state=None):
        self.test_ratio = test_ratio
        self.n_folds = n_folds
        self.random_state = random_state

    def _gen_samples(self, y):
        sss = StratifiedShuffleSplit(n_splits=self.n_folds, test_size=self.test_ratio, random_state=self.random_state)
        for i, (train, test) in enumerate(
            sss.split(X=np.zeros(len(y)), y=y)
        ):  # np.zeros used as a placeholder for X
            # print('test', test)
            yield i, train, test

    @property
    def name(self):
        return self.__repr__()

    def __repr__(self):
        return f"<{self.__class__.__name__} with {self.n_folds} folds and using test_ratio {self.test_ratio}>"


class BalancedStratifiedRatioSampler(Sampler):
    """
    This sampler can e.g. be used if equal size sampling (subsampling of the majority class)
    is needed in aggregated conformal predictors

    Parameters
    ----------
    test_ratio : float
        Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split
        Note: according to sklearn, test_ratio could also be int or None. To allow this, we would need to remove
              the assert statement
    n_folds : int
        Number of re-shuffling and splitting iterations.

    Attributes
    ----------
    test_ratio : float
        Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split
        Note: according to sklearn, test_ratio could also be int or None. To allow this, we would need to remove
              the assert statement
    n_folds : int
        Number of re-shuffling and splitting iterations.

    Examples
    --------
    todo
    """

    def __init__(self, test_ratio=0.3, n_folds=1, random_state=None):
        self.test_ratio = test_ratio
        self.n_folds = n_folds
        self.random_state = random_state

    def _gen_samples(self, y):
        sss = StratifiedShuffleSplit(n_splits=self.n_folds, test_size=self.test_ratio, random_state=self.random_state)
        for i, (train, test) in enumerate(
            sss.split(X=np.zeros(len(y)), y=y)
        ):  # np.zeros used as a placeholder for X
            # train and test are balanced separately. Balancing them before splitting would mean that
            # we use the same balanced dataset (same majority class instances removed) in every step
            y_train, y_test = y[train], y[test]
            test = self._balance(y_test, test, ratio=1.0)
            train = self._balance(y_train, train, ratio=1.0)

            yield i, train, test

    @property
    def name(self):
        return self.__repr__()

    def __repr__(self):
        return f"<{self.__class__.__name__} with {self.n_folds} folds and using test_ratio {self.test_ratio}>"


class CrossValidationSampler(Sampler):
    """
    This is a sampler to be used for crossvalidation or cross-conformal predictors (not implemented yet)

    Parameters
    ----------
    n_folds : int
        Number of folds. Must be at least 2

    Attributes
    ----------
    n_folds : int
        Number of folds. Must be at least 2

    Examples
    --------
    todo
    """

    def __init__(self, n_folds=5, random_state=None):
        self.n_folds = n_folds
        self.random_state = random_state

    def _gen_samples(self, y):
        folds = StratifiedKFold(n_splits=self.n_folds, random_state=self.random_state, shuffle=True)
        for i, (train, test) in enumerate(folds.split(X=np.zeros(len(y)), y=y)):
            # i specifies the fold of the crossvalidation, i.e. between 0 and 4
            yield i, train, test

    @property
    def name(self):
        return self.__repr__()

    def __repr__(self):
        return f"<{self.__class__.__name__} with {self.n_folds} folds>"


# --------------------------------
# Inductive Conformal Predictor
# --------------------------------


class InductiveConformalPredictor(IcpClassifier):
    """
    Inductive Conformal Prediction Classifier
    This is a subclass of the IcpClassifier from nonconformist
    https://github.com/donlnz/nonconformist/blob/master/nonconformist/icp.py
    The subclass allows to further extend the class to the needs of this project

    Parameters
    ----------
    # Note: some of the parameters descriptions are copied from nonconformist IcpClassifier

    condition: condition for calculating p-values. Default condition is mondrian (calibration with 1 list
     of nc scores per class).
     Note that default condition in nonconformist is 'lambda x: 0' (only one list for both/multiple classes (?)
     For mondrian condition, see: https://pubs.acs.org/doi/10.1021/acs.jcim.7b00159

    nc_function : BaseScorer
        Nonconformity scorer object used to calculate nonconformity of
        calibration examples and test patterns. Should implement ``fit(x, y)``
        and ``calc_nc(x, y)``.

    mondrian: bool
        define whether to use mondrian condition or not (see: https://pubs.acs.org/doi/10.1021/acs.jcim.7b00159)

    Attributes
    ----------
    # Note: some of the attributes descriptions are copied from nonconformist IcpClassifier

    condition: condition for calculating p-values. Note that if we want to use 'mondrian' condition,
    we can either input condition='mondrian' or condition=(lambda instance: instance[1]). Then, the condition.name will
     be saved, which is useful for serialisation

    cal_x : numpy array of shape [n_cal_examples, n_features]
        Inputs of calibration set.
    cal_y : numpy array of shape [n_cal_examples]
        Outputs of calibration set.
    nc_function : BaseScorer
        Nonconformity scorer object used to calculate nonconformity scores.
    classes : numpy array of shape [n_classes]
        List of class labels, with indices corresponding to output columns
        of IcpClassifier.predict()

    Examples
    --------
    todo
    """

    def __init__(self, nc_function, smoothing=False, condition=None, random_state=None):
        super().__init__(nc_function, condition=condition, smoothing=smoothing)
        self.random_state = random_state

    def predict(self, x, significance=None):
        """Predict the output values for a set of input patterns.
        Parameters
        ----------
        x : numpy array of shape [n_samples, n_features]
            Inputs of patters for which to predict output values.
        significance : float or None
            Significance level (maximum allowed error rate) of predictions.
            Should be a float between 0 and 1. If ``None``, then the p-values
            are output rather than the predictions.
        Returns
        -------
        p : numpy array of shape [n_samples, n_classes]
            If significance is ``None``, then p contains the p-values for each
            sample-class pair; if significance is a float between 0 and 1, then
            p is a boolean array denoting which labels are included in the
            prediction sets.
        """
        # TODO: if x == self.last_x ...
        # fixme: When predicting whole datasets, the value from random.uniform in smoothing (calc_p) will change.
        # however, will it be a problem when single instances are predicted?
        np.random.seed(self.random_state)
        n_test_objects = x.shape[0]
        p = np.zeros((n_test_objects, self.classes.size))

        ncal_ngt_neq = self._get_stats(x)

        for i in range(len(self.classes)):
            for j in range(n_test_objects):
                p[j, i] = calc_p(ncal_ngt_neq[j, i, 0],
                                 ncal_ngt_neq[j, i, 1],
                                 ncal_ngt_neq[j, i, 2],
                                 self.smoothing)

        if significance is not None:
            return p > significance
        else:
            return p

    def _get_stats(self, x):
        """
        This method is copied from nonconformist.  On the cluster, I got the error
        "AttributeError: 'InductiveConformalPredictor' object has no attribute '_get_stats'"
        After copying the method into my code, the error was gone

        Parameters
        ----------
        x

        Returns
        -------

        """
        n_test_objects = x.shape[0]
        ncal_ngt_neq = np.zeros((n_test_objects, self.classes.size, 3))
        for i, c in enumerate(self.classes):
            test_class = np.zeros(x.shape[0], dtype=self.classes.dtype)
            test_class.fill(c)

            # TODO: maybe calculate p-values using cython or similar
            # TODO: interpolated p-values

            # TODO: nc_function.calc_nc should take X * {y1, y2, ... ,yn}
            test_nc_scores = self.nc_function.score(x, test_class)
            for j, nc in enumerate(test_nc_scores):
                cal_scores = self.cal_scores[self.condition((x[j, :], c))][::-1]
                n_cal = cal_scores.size

                idx_left = np.searchsorted(cal_scores, nc, 'left')
                idx_right = np.searchsorted(cal_scores, nc, 'right')

                ncal_ngt_neq[j, i, 0] = n_cal
                ncal_ngt_neq[j, i, 1] = n_cal - idx_right
                ncal_ngt_neq[j, i, 2] = idx_right - idx_left

        return ncal_ngt_neq


# -------------------------------
# Conformal Predictor Aggregators
# -------------------------------


class BaseConformalPredictorAggregator:
    """
    Combines multiple InductiveConformalPredictor predictors into an aggregated model
    The structure of this class is adapted from the nonconformist acp module:
    https://github.com/donlnz/nonconformist/blob/master/nonconformist/acp.py

    Parameters
    ----------
    predictor : object
        Prototype conformal predictor (i.e. InductiveConformalPredictor)
        used for defining conformal predictors included in the aggregate model.

    Attributes
    ----------
    predictor : object
        Prototype conformal predictor (i.e. InductiveConformalPredictor)
        used for defining conformal predictors included in the aggregate model.
    """

    def __init__(self, predictor):
        self.predictor = predictor

    def _fit_calibrate(self, **kwargs):
        raise NotImplementedError("Implement in your subclass")

    def fit_calibrate(self, **kwargs):
        return self._fit_calibrate(**kwargs)

    def _predict(self, **kwargs):
        raise NotImplementedError("Implement in your subclass")

    def predict(self, **kwargs):
        return self._predict(**kwargs)

    @property
    def name(self):
        raise NotImplementedError("Implement in your subclass")


class AggregatedConformalPredictor(BaseConformalPredictorAggregator):
    """
    Generates an aggregated conformal predictor (acp) from multiple InductiveConformalPredictor predictors
    The structure of this class is adapted from the nonconformist acp module:
    https://github.com/donlnz/nonconformist/blob/master/nonconformist/acp.py

    Parameters
    ----------
    predictor : object
        Prototype conformal predictor (i.e. InductiveConformalPredictor)
        used for defining conformal predictors included in the aggregate model.
    sampler : object
        Sampler object used to generate training and calibration examples
        for the underlying conformal predictors.
    aggregation_func : callable
        Function used to aggregate the predictions of the underlying
        conformal predictors. Defaults to ``numpy.median``.
    n_models : int
        Number of models to aggregate.

    Attributes
    ----------
    predictor : object
        Prototype conformal predictor (i.e. InductiveConformalPredictor)
        used for defining conformal predictors included in the aggregate model.
    sampler : object
        Sampler object used to generate training and calibration examples
        for the underlying conformal predictors.
    agg_func : callable
        Function used to aggregate the predictions of the underlying
        conformal predictors. Defaults to ``numpy.median``.
    n_models : int
        Number of models to aggregate.
    predictors_fitted : list
        contains fitted ICP's
    predictors_calibrated : list
        contains calibrated ICP's
    predictors_calibrated_update : list
        contains fitted ICP's calibrated with the update dataset

    Examples
    --------
    todo
    """

    def __init__(self, predictor, sampler, aggregation_func=None):
        super().__init__(predictor)
        self.predictor = predictor
        self.predictors_fitted = []
        self.predictors_calibrated = []

        self.sampler = sampler
        self.n_models = sampler.n_folds
        if aggregation_func is not None:
            self.agg_func = aggregation_func
        else:
            self.agg_func = np.median

    def _fit_calibrate(
            self,
            X_train=None,
            y_train=None,
    ):

        self.predictors_fitted.clear()
        self.predictors_calibrated.clear()

        samples = self.sampler.gen_samples(labels=y_train)
        for loop, p_train, cal in samples:
            predictor = copy.deepcopy(
                self.predictor
            )

            # Fit
            predictor.train_index = p_train
            predictor.fit(X_train[p_train, :], y_train[p_train])
            self.predictors_fitted.append(predictor)

            # Calibrate
            predictor_calibration = copy.deepcopy(predictor)
            predictor_calibration.calibrate(X_train[cal, :], y_train[cal])
            self.predictors_calibrated.append(predictor_calibration)

    @staticmethod
    def _f(predictor, X):
        return predictor.predict(X, None)

    def _predict(self, X_score=None):
        predictions = np.dstack(
            [self._f(p, X_score) for p in self.predictors_calibrated]
        )
        predictions = self.agg_func(predictions, axis=2)

        return predictions

    @property
    def name(self):
        return self.__repr__()

    def __repr__(self):
        return f"<{self.__class__.__name__}, samples generated with {self.sampler}, {self.n_models} models built>"


class ContinuousCalibrationAggregatedConformalPredictor(AggregatedConformalPredictor):
    """
    An aggregated conformal predictor class, specificly adapted for continuous calibration
    """

    def __init__(self, predictor, sampler, aggregation_func=None):
        super().__init__(predictor, sampler, aggregation_func)
        self.predictors_calibrated_update = {}

    def _fit_calibrate(
        self, X_train=None, y_train=None,
    ):
        # todo: do we want to save known labels within this method, i.e. y_score?
        # todo: are indices of p_train and cal saved - do we need to save them??

        self.predictors_fitted.clear()
        self.predictors_calibrated.clear()

        samples = self.sampler.gen_samples(labels=y_train)
        for loop, p_train, cal in samples:  # e.g. 20 loops
            predictor = copy.deepcopy(self.predictor)

            # fit
            predictor.train_index = p_train
            predictor.fit(X_train[p_train, :], y_train[p_train])
            self.predictors_fitted.append(predictor)

            # calibrate (no update)
            predictor_calibration = copy.deepcopy(predictor)
            predictor_calibration.calibrate(X_train[cal, :], y_train[cal])
            self.predictors_calibrated.append(predictor_calibration)

    def calibrate_update(self, X_update, y_update):
        n = 1
        while f"update_{n}" in self.predictors_calibrated_update:
            print('n', n)
            n += 1
        print('final n: ', n)
        self.predictors_calibrated_update[f"update_{n}"] = []
        for predictor_fitted in self.predictors_fitted:
            predictor_calibration_update = copy.deepcopy(predictor_fitted)
            predictor_calibration_update.calibrate(X_update, y_update)
            self.predictors_calibrated_update[f"update_{n}"].append(
                predictor_calibration_update
            )
        return n

    def _f(self, predictor, X):
        return predictor.predict(X, None)

    def _predict(self, X_score=None):
        predictions = np.dstack(
            [self._f(p, X_score) for p in self.predictors_calibrated]
        )
        predictions = self.agg_func(predictions, axis=2)
        return predictions

    def predict_calibrate_update(self, updated_number, X_score=None):
        print('updated_number: ', updated_number)
        predictions = np.dstack(
            [
                self._f(p, X_score)
                for p in self.predictors_calibrated_update[f"update_{updated_number}"]
            ]
        )
        predictions = self.agg_func(predictions, axis=2)
        return predictions


# --------------------------------
# Crossvalidation
# --------------------------------


class CrossValidator:
    def __init__(self, predictor, cv_splitter):
        self.sampler = cv_splitter
        self.predictor = predictor
        self._evaluation_dfs = {}
        self._predictions = {}
        self.predictors = None
        self.num_actives = 0
        self.num_inactives = 0

        self.train_indices = []
        self.test_indices = []

        self._score_names = None
        self._cv_names = None

    def cross_validate(
        self,
        steps,
        endpoint,
        X_train,
        y_train,
        X_score,
        y_score,
        class_wise_evaluation=False,
    ):

        num_actives = y_train.sum()
        self.num_actives = num_actives
        self.num_inactives = len(y_train) - num_actives

        cv_predictions = []
        pred_score_predictions = []
        cv_y_test = []
        predictors = []

        cv_evaluations = self._create_empty_evaluations_dict()
        pred_score_evaluations = self._create_empty_evaluations_dict()

        samples = self.sampler.gen_samples(labels=y_train)

        for fold, train, test in samples:
            print('fold: ', fold)
            self.train_indices.append(list(train))
            self.test_indices.append(list(test))

            cv_y_test.append(y_train[test])

            # ----------------------------------------------------------
            # Fit and calibrate ACP
            # ----------------------------------------------------------

            predictor = copy.deepcopy(self.predictor)

            predictor.fit_calibrate(X_train=X_train[train], y_train=y_train[train])
            predictors.append(predictor)

            # ----------------------------------------------------------
            # Make predictions with ACP
            # ----------------------------------------------------------

            # CV prediction (internal CV test set)
            cv_prediction = predictor.predict(X_score=X_train[test])
            cv_predictions.append(cv_prediction)

            # Predict (external) score set using predictor with and without updated calibration set
            pred_score_prediction = predictor.predict(X_score=X_score)
            pred_score_predictions.append(pred_score_prediction)

            cv_evaluations = self._evaluate(
                cv_prediction,
                y_train[test],
                cv_evaluations,
                endpoint,
                fold=fold,
                steps=steps,
                class_wise=class_wise_evaluation,
            )

            pred_score_evaluations = self._evaluate(
                pred_score_prediction,
                y_score,
                pred_score_evaluations,
                endpoint,
                fold=fold,
                steps=steps,
                class_wise=class_wise_evaluation,
            )
        print("YSCORE", type(y_score), "++++++++++++++++++++++++++++++++")
        print(type(cv_y_test[0]))
        self._evaluation_dfs["cv"] = pd.DataFrame(cv_evaluations)
        self._evaluation_dfs["pred_score"] = pd.DataFrame(pred_score_evaluations)

        self._predictions["cv"] = [cv_predictions, cv_y_test]
        self._predictions["pred_score"] = [pred_score_predictions, np.tile(y_score, ((fold+1), 1))]
        print(type(self._predictions["pred_score"][1]))
        print(self._predictions["pred_score"][1])

        # fixme: is len y_score ok or do need n times y_score?
        self.predictors = predictors
        return pd.DataFrame(cv_evaluations)

    def cross_validate_calibrate_update(
            self, steps, endpoint, X_update, y_update, X_score, y_score,
            class_wise_evaluation=False
    ):
        assert self.predictors is not None
        predictors = copy.deepcopy(self.predictors)
        cal_update_predictors = []
        predictions = []
        evaluations = self._create_empty_evaluations_dict()

        for fold, predictor in enumerate(predictors):
            print('fold: ', fold)
            n = predictor.calibrate_update(X_update, y_update)
            cal_update_predictors.append(predictor)
            prediction = predictor.predict_calibrate_update(n, X_score=X_score)
            predictions.append(prediction)

            evaluations = self._evaluate(
                prediction,
                y_score,
                evaluations,
                endpoint,
                fold=fold,
                steps=steps,
                class_wise=class_wise_evaluation,
            )
        self._evaluation_dfs[f"cal_update_{n}"] = pd.DataFrame(evaluations)
        self._predictions[f"cal_update_{n}"] = [predictions, np.tile(y_score, ((fold+1), 1))]
        self.predictors = cal_update_predictors

    @staticmethod
    def _create_empty_evaluations_dict():

        evaluation_measures = [
            "validity",
            "validity_0",
            "validity_1",
            "validity_bal",
            "error_rate",
            "error_rate_0",
            "error_rate_1",
            "error_rate_bal",
            "efficiency",
            "efficiency_0",
            "efficiency_1",
            "efficiency_bal",
            "accuracy",
            "accuracy_0",
            "accuracy_1",
            "accuracy_bal",
        ]

        empty_evaluations_dict = {}
        for measure in evaluation_measures:
            empty_evaluations_dict[measure] = []

        empty_evaluations_dict["significance_level"] = []
        empty_evaluations_dict["fold"] = []

        return empty_evaluations_dict

    @staticmethod
    def _evaluate(
        prediction, y_true, evaluations, endpoint, fold, steps, class_wise=True
    ):
        # print("YTRUE", y_true, " ==================================== ")
        # fixme later 1: currently class-wise evaluation measures are calculated anyways but only saved
        #  if class_wise is True. Library might be changed, so that they are only calculated if necessary
        # fixme later 2: validity and error_rate could be calculated using the same method, no need to do this twice
        evaluator = Evaluator(prediction, y_true, endpoint)
        sl = [i / float(steps) for i in range(steps)] + [1]

        validities_list = ["validity", "validity_0", "validity_1", "validity_bal"]
        error_rates_list = ["error_rate", "error_rate_0", "error_rate_1", "error_rate_bal"]
        efficiencies_list = ["efficiency", "efficiency_0", "efficiency_1", "efficiency_bal"]
        accuracies_list = ["accuracy", "accuracy_0", "accuracy_1", "accuracy_bal"]

        validities = [
            evaluator.calculate_validity(i / float(steps)) for i in range(steps)
        ] + [evaluator.calculate_validity(1)]
        for validity in validities_list:
            evaluations[validity].extend([val[validity] for val in validities])

        error_rates = [
            evaluator.calculate_error_rate(i / float(steps)) for i in range(steps)
        ] + [evaluator.calculate_error_rate(1)]
        for error_rate in error_rates_list:
            evaluations[error_rate].extend([err[error_rate] for err in error_rates])

        efficiencies = [
            evaluator.calculate_efficiency(i / float(steps)) for i in range(steps)
        ] + [evaluator.calculate_efficiency(1)]
        for efficiency in efficiencies_list:
            evaluations[efficiency].extend([eff[efficiency] for eff in efficiencies])
        accuracies = [
            evaluator.calculate_accuracy(i / float(steps)) for i in range(steps)
        ] + [evaluator.calculate_accuracy(1)]
        for accuracy in accuracies_list:
            evaluations[accuracy].extend([acc[accuracy] for acc in accuracies])

        evaluations["significance_level"].extend(sl)
        evaluations["fold"].extend([fold] * (steps + 1))
        return evaluations

    @property
    def averaged_evaluation_df_cv(self):
        return self._average_evaluation_df(
            self._evaluation_dfs["cv"], self.num_actives, self.num_inactives
        )

    @property
    def averaged_evaluation_df_pred_score(self):
        return self._average_evaluation_df(
            self._evaluation_dfs["pred_score"], self.num_actives, self.num_inactives
        )

    @property
    def averaged_evaluation_df_cal_update_1(self):
        assert "cal_update_1" in self._evaluation_dfs.keys()
        return self._average_evaluation_df(
            self._evaluation_dfs["cal_update_1"], self.num_actives, self.num_inactives
        )

    @property
    def averaged_evaluation_df_cal_update_2(self):
        assert "cal_update_2" in self._evaluation_dfs.keys()
        return self._average_evaluation_df(
            self._evaluation_dfs["cal_update_2"], self.num_actives, self.num_inactives
        )

    @property
    def averaged_evaluation_df_cal_update_3(self):
        assert "cal_update_3" in self._evaluation_dfs.keys()
        return self._average_evaluation_df(
            self._evaluation_dfs["cal_update_3"], self.num_actives, self.num_inactives
        )

    @property
    def averaged_evaluation_df_cal_update_4(self):
        assert "cal_update_4" in self._evaluation_dfs.keys()
        return self._average_evaluation_df(
            self._evaluation_dfs["cal_update_4"], self.num_actives, self.num_inactives
        )

    @property
    def averaged_evaluation_df_cal_update_5(self):
        assert "cal_update_5" in self._evaluation_dfs.keys()
        return self._average_evaluation_df(
            self._evaluation_dfs["cal_update_5"], self.num_actives, self.num_inactives
        )

    @staticmethod
    def _average_evaluation_df(evaluation_df, num_actives, num_inactives):
        evaluation_df_grouped = evaluation_df.groupby(
            by="significance_level"
        ).aggregate([np.mean, np.std])
        evaluation_df_grouped.drop(["fold"], axis=1, inplace=True)
        evaluation_df_grouped.columns = [
            " ".join((a, b)) for a, b in evaluation_df_grouped.columns
        ]
        evaluation_df_grouped.columns = evaluation_df_grouped.columns.get_level_values(
            0
        )
        evaluation_df_grouped["significance_level"] = evaluation_df_grouped.index
        evaluation_df_grouped["num_actives"] = num_actives
        evaluation_df_grouped["num_inactives"] = num_inactives
        return evaluation_df_grouped

    @property
    def cv_predictions_df(self):
        return self._format_predictions_df(self._predictions["cv"], self._cv_names)

    @property
    def pred_score_predictions_df(self):
        return self._format_predictions_df(
            self._predictions["pred_score"], self._score_names
        )

    @property
    def cal_update_1_predictions_df(self):
        return self._format_predictions_df(
            self._predictions["cal_update_1"], self._score_names
        )

    @property
    def cal_update_2_predictions_df(self):
        return self._format_predictions_df(
            self._predictions["cal_update_2"], self._score_names
        )

    @property
    def cal_update_3_predictions_df(self):
        return self._format_predictions_df(
            self._predictions["cal_update_3"], self._score_names
        )

    @property
    def cal_update_4_predictions_df(self):
        return self._format_predictions_df(
            self._predictions["cal_update_4"], self._score_names
        )

    @property
    def cal_update_5_predictions_df(self):
        return self._format_predictions_df(
            self._predictions["cal_update_5"], self._score_names
        )

    @staticmethod
    def _format_predictions_df(predictions, names):
        print("FORMATTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(len(predictions))
        # print((predictions[0]))
        print("1", type(predictions[1]), (predictions[1]))
        pred_dfs = []
        for i, pred in enumerate(predictions[0]):
            print(i)
            pred_df = pd.DataFrame(data=predictions[0][i])
            pred_df["true"] = predictions[1][i]
            if names is not None:
                pred_df["Name"] = names
            pred_dfs.append(pred_df)
        return pd.concat(pred_dfs)

    def calibration_plot(
        self,
        averaged_evaluation_df,
        endpoint,
        colours=("blue", "darkred", "deepskyblue", "lightcoral"),
        class_wise=True,
        efficiency=True,
        **kwargs,
    ):

        return self._calibration_plot(
            averaged_evaluation_df=averaged_evaluation_df,
            endpoint=endpoint,
            colours=colours,
            class_wise=class_wise,
            efficiency=efficiency,
        )

    @staticmethod
    def _calibration_plot(
        averaged_evaluation_df,
        endpoint,
        colours=("blue", "darkred", "deepskyblue", "lightcoral"),
        class_wise=True,
        efficiency=True,
    ):

        # print(self._evaluation_df.shape, 'fixme: assert that correct shape!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # assert self.evaluation_df.shape ==

        if class_wise and efficiency:
            evaluation_measures = [
                "error_rate_0",
                "error_rate_1",
                "efficiency_0",
                "efficiency_1",
            ]
        elif class_wise and not efficiency:
            evaluation_measures = ["error_rate_0", "error_rate_1"]

        elif not class_wise and efficiency:
            evaluation_measures = ["error_rate", "efficiency"]

        else:  # not class_wise and not efficiency
            evaluation_measures = ["error_rate"]

        # averaged_evaluation_df = averaged_evaluation_df
        # print(averaged_evaluation_df)

        fig, ax = plt.subplots()

        ax.plot([0, 1], [0, 1], "--", linewidth=1, color="black")
        sl = averaged_evaluation_df["significance_level"]

        for ev, colour in zip(evaluation_measures, colours):
            ev_mean = averaged_evaluation_df[f"{ev} mean"]
            ev_std = averaged_evaluation_df[f"{ev} std"]
            ax.plot(sl, ev_mean, label=True, c=colour)
            ax.fill_between(
                sl, ev_mean - ev_std, ev_mean + ev_std, alpha=0.3, color=colour
            )
        #
        major_ticks = np.arange(0, 101, 20)
        minor_ticks = np.arange(0, 101, 5)
        #
        ax.set_xticks(minor_ticks / 100.0, minor=True)
        ax.set_yticks(major_ticks / 100.0)
        ax.set_yticks(minor_ticks / 100.0, minor=True)

        # ax.set_title(endpoint, fontsize=16)
        ax.grid(which="minor", linewidth=0.5)  # alpha=0.5)
        ax.grid(which="major", linewidth=1.5)  # alpha=0.9, linewidth=2.0)

        ax.set_xlabel("significance",)
        ax.set_ylabel("error rate")
        eval_legend = evaluation_measures.copy()
        eval_legend.insert(0, "expected_error_rate")
        fig.legend(eval_legend, bbox_to_anchor=(1.25, 0.75))
        plt.title(endpoint)
        # plt.show()
        return plt
        # plt.savefig(f'../../data/poster_plots/predict_score_{ep}.png')


# --------------------------------
# Evaluator
# --------------------------------


class Evaluator:
    def __init__(self, y_pred, y_true=None, score_set=None, endpoint=None):
        if y_true is None:
            y_true = score_set.measurements[endpoint]
        # print(y_pred, type(y_pred))
        y_pred_0 = y_pred[:, 0]
        y_pred_1 = y_pred[:, 1]
        _prediction_df = pd.DataFrame(
            data={"p0": y_pred_0, "p1": y_pred_1, "known_label": y_true}
        )
        # print(_prediction_df.shape)
        _prediction_df = (
            _prediction_df.dropna()
        )  # fixme: is this necessary? We only consider values == 0.0 and
        # values == 1.0 anyways
        # print(_prediction_df.shape)
        self._prediction_df = _prediction_df
        self.endpoint = endpoint

    def _calculate_set_sizes(self):
        nof_neg = float(sum(self._prediction_df["known_label"].values == 0.0))
        nof_pos = float(sum(self._prediction_df["known_label"].values == 1.0))
        nof_all = float(nof_neg + nof_pos)

        return nof_all, nof_neg, nof_pos

    def _calculate_nof_one_class_predictions(self, label, significance):
        """
        Calculate number of one class predictions for a specific class at a given significance level
        """

        # Get number of compounds that have respective label
        # and only one of the p-values fullfills significance level
        nof = sum(
            (self._prediction_df["known_label"].values == label)
            & (
                (
                    (self._prediction_df.p0.values < significance)
                    & (self._prediction_df.p1.values >= significance)
                )
                | (
                    (self._prediction_df.p0.values >= significance)
                    & (self._prediction_df.p1.values < significance)
                )
            )
        )

        return nof

    def calculate_efficiency(self, significance):
        """
           Calculate ratio of efficient predictions, i.e. prediction sets containig one single label
           """

        # Calculate total number of compounds, class-wise and all compounds
        total, total_0, total_1 = self._calculate_set_sizes()

        # Calculate number of efficiently predicted compounds
        # (only one label not in prediction set at given significance level)
        # class-wise
        efficiency_0 = self._calculate_nof_one_class_predictions(0.0, significance)
        efficiency_1 = self._calculate_nof_one_class_predictions(1.0, significance)

        # Calculate efficiency rate, class-wise and for all compounds
        efficiency_rate_0 = round(efficiency_0 / total_0, 3)
        efficiency_rate_1 = round(efficiency_1 / total_1, 3)
        efficiency_rate = round(((efficiency_0 + efficiency_1) / total), 3)

        return {
            "efficiency": efficiency_rate,
            "efficiency_0": efficiency_rate_0,
            "efficiency_1": efficiency_rate_1,
            "efficiency_bal": (efficiency_rate_1 + efficiency_rate_0)/2.
        }

    def calculate_validity(self, significance):
        """
           Calculate ratio of valid predictions, i.e. prediction sets containing the correct label
           """

        # Calculate total number of compounds, class-wise and all compounds
        total, total_0, total_1 = self._calculate_set_sizes()

        # Calculate number of wrongly predicted compounds
        # (correct label not in prediction set at given significance level)
        # class-wise
        error_0 = sum(
            (self._prediction_df["known_label"].values == 0.0)
            & (self._prediction_df.p0.values < significance)
        )
        error_1 = sum(
            (self._prediction_df["known_label"].values == 1.0)
            & (self._prediction_df.p1.values < significance)
        )

        # Calculate error rate, class-wise and for all compounds
        error_rate_0 = round(error_0 / total_0, 3)
        error_rate_1 = round(error_1 / total_1, 3)
        error_rate = round(((error_0 + error_1) / total), 3)

        return {
            "validity": (1 - error_rate),
            "validity_0": (1 - error_rate_0),
            "validity_1": (1 - error_rate_1),
            "validity_bal": ((1 - error_rate_0) + (1 - error_rate_1))/2.
        }

    def calculate_error_rate(self, significance):
        """
           Calculate ratio of valid predictions, i.e. prediction sets containing the correct label
           """

        # Calculate total number of compounds, class-wise and all compounds
        total, total_0, total_1 = self._calculate_set_sizes()

        # Calculate number of wrongly predicted compounds
        # (correct label not in prediction set at given significance level)
        # class-wise
        error_0 = sum(
            (self._prediction_df["known_label"].values == 0.0)
            & (self._prediction_df.p0.values < significance)
        )
        error_1 = sum(
            (self._prediction_df["known_label"].values == 1.0)
            & (self._prediction_df.p1.values < significance)
        )

        # Calculate error rate, class-wise and for all compounds
        error_rate_0 = round(error_0 / total_0, 3)
        error_rate_1 = round(error_1 / total_1, 3)
        error_rate = round(((error_0 + error_1) / total), 3)

        return {
            "error_rate": error_rate,
            "error_rate_0": error_rate_0,
            "error_rate_1": error_rate_1,
            "error_rate_bal": (error_rate_0 + error_rate_1)/2.
        }

    def calculate_accuracy(self, significance):
        """
          Calculate ratio of accurate predictions, i.e. efficient prediction sets containing the one correct label
          """
        # Calculate number of efficiently predicted compounds
        # (only one label not in prediction set at given significance level)
        # class-wise
        efficiency_0 = self._calculate_nof_one_class_predictions(0.0, significance)
        efficiency_1 = self._calculate_nof_one_class_predictions(1.0, significance)
        efficiency = efficiency_0 + efficiency_1

        # Calculate number of correctly and efficiently predicted compounds
        # (only one correct label in prediction set at given significance level)
        # class-wise
        accuracy_0 = sum(
            (self._prediction_df["known_label"].values == 0.0)
            & (self._prediction_df.p0.values >= significance)
            & (self._prediction_df.p1.values < significance)
        )
        accuracy_1 = sum(
            (self._prediction_df["known_label"].values == 1.0)
            & (self._prediction_df.p0.values < significance)
            & (self._prediction_df.p1.values >= significance)
        )

        # Calculate accuracy rate, class-wise and for all compounds
        # todo: how to handle division by zero??
        accuracy_rate_0 = (
            round(accuracy_0 / efficiency_0, 3) if efficiency_0 != 0 else 0
        )
        accuracy_rate_1 = (
            round(accuracy_1 / efficiency_1, 3) if efficiency_1 != 0 else 0
        )
        accuracy_rate = (
            round(((accuracy_0 + accuracy_1) / efficiency), 3) if efficiency != 0 else 0
        )

        return {
            "accuracy": accuracy_rate,
            "accuracy_0": accuracy_rate_0,
            "accuracy_1": accuracy_rate_1,
            "accuracy_bal": (accuracy_rate_0 + accuracy_rate_1)/2.
        }

    def calibration_plot(self, steps):
        # fixme: I am not sure yet, if this method should live here
        # todo: include class-wise evaluation
        validities_tot = [
            self.calculate_validity(i / float(steps))["validity"] for i in range(steps)
        ] + [self.calculate_validity(1)["validity"]]
        error_rate_tot = [1 - i for i in validities_tot]
        efficiencies_tot = [
            self.calculate_efficiency(i / float(steps))["efficiency"]
            for i in range(steps)
        ] + [self.calculate_efficiency(1)["efficiency"]]
        sl = [i / float(steps) for i in range(steps)] + [1]
        # print(sl)

        fig, ax = plt.subplots()

        ax.plot([0, 1], [0, 1], "--", linewidth=1, color="black")
        ax.plot(sl, error_rate_tot, label=self.endpoint)
        ax.plot(sl, efficiencies_tot)
        ax.legend(loc="lower right")
        ax.xlabel("significance")
        ax.ylabel("error rate")
        return fig


def calc_p(ncal, ngt, neq, smoothing=False, random_state=None):
    """
    This function was taken and adapted from nonconformist.utils. Adaption was necessary to allow setting a
    random state for smoothing (np.random.uniform)

    Parameters
    ----------
    random_state
    ncal
    ngt
    neq
    smoothing

    Returns
    -------

    """

    if smoothing:
        return (ngt + (neq + 1) * np.random.uniform(0, 1)) / (ncal + 1)
    else:
        return (ngt + neq + 1) / (ncal + 1)