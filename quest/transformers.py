import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer, make_column_selector

from scipy.stats import zscore
from scipy.optimize import minimize

from qiskit_optimization import QuadraticProgram

from .utils import logger

_logger = logger(__name__)


class QuboPreprocessor(BaseEstimator, TransformerMixin):
    """
    QuboPreprocessor is transforming data into the right shape for the QUBO Problem.
    """
    def __init__(self, quantiles: int = 4):
        self.quantiles = quantiles

    @property
    def get_feature_names(self) -> list:
        """
        Retrieve feature names after transformation.
        """
        return [col.split("__")[1] for col in self.transformer_.get_feature_names_out()]

    def fit(self, X: np.array):
        """
        Here, fitting means binning numeric data and one-hot-encoding categorical data.

        Args:
            X: array to be transformed
        """
        self.transformer_ = make_column_transformer(
                        (KBinsDiscretizer(n_bins=self.quantiles, encode='onehot-dense',  strategy='quantile'),
                         make_column_selector(dtype_include=np.number)),
                        (OneHotEncoder(sparse_output=False,  handle_unknown="infrequent_if_exist"),
                         make_column_selector(dtype_include=[object, 'category']))
        ).fit(X)

        return self

    def transform(self, X: np.array):
        """
        Transform method for unseen data

        Args:
            X: array to be transformed
        """
        X = self.transformer_.transform(X)
        return pd.DataFrame(X, columns=self.get_feature_names)


class QuboTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, coeff_cap: float = 1/2, max_features: int = 100):
        self.coeff_cap = coeff_cap
        self.max_features = max_features

    def _data_to_corr(self, features: np.ndarray, labels: np.ndarray) -> (pd.DataFrame, pd.DataFrame):

        # data = pd.get_dummies(features)
        if not isinstance(labels, pd.DataFrame):
            data = pd.DataFrame(features)
        # data = data.apply(lambda x: zscore(x))
        # data = pd.DataFrame(MinMaxScaler().fit_transform(data), columns=features.columns)
        else:
            data = features
        data_label = data.copy()
        if isinstance(labels, pd.DataFrame) or isinstance(labels, pd.Series):
            data_label['label'] = labels.values
        else:
            data_label['label'] = labels

        corr_to_target = pd.DataFrame(
            data_label.corr()['label'].abs().sort_values(ascending=False).drop(['label'])[:self.max_features]
        )
        # na_index = corr_to_target.loc[corr_to_target.label.isna()].index

        corr_to_target = corr_to_target.dropna()
        self.pre_selected_features = corr_to_target.index
        corr_features = data.loc[:, self.pre_selected_features].corr()

        return corr_to_target, corr_features

    @staticmethod
    def _q_to_qp(corr_to_target: pd.DataFrame, corr_features: pd.DataFrame, coeff_cap: float = 1/2) -> QuadraticProgram:
        # -- Overall Params
        qp_name = 'QUEST'

        # --------------
        var_selected = []
        var_tot = [var for i, var in zip(range(len(corr_to_target.index)), corr_to_target.index)]

        def corr_target_func(field_i):
            return round(corr_to_target.loc[field_i].values[0], 6)

        def corr_func(field_i, field_j):
            return round(corr_features.loc[field_i, field_j], 6)

        def index_to_field(i):
            return corr_to_target.index[i]

        # Define the qp (quadratic problem)
        qp = QuadraticProgram(qp_name)
        for i in range(len(var_tot)):
            qp.binary_var(name="a" + str(i))  # these are the a_i
        linear_terms = {"a" + str(i): abs(corr_target_func(i_field)) for i, i_field in
                        zip(range(len(var_tot)), var_tot)}

        # Note that we apply here the abs(...) of the correlation among featuers.
        quadratic_terms = {("a" + str(i), "a" + str(j)):
                               - coeff_cap * abs(corr_func(index_to_field(i), index_to_field(j)))
                           for i in range(len(var_tot))
                           for j in range(len(var_tot))
                           if i < j}

        qp.maximize(constant=0, linear=linear_terms, quadratic=quadratic_terms)

        return qp

    def fit(self, X, y=None):

        corr_to_target, corr_features = self._data_to_corr(features=X, labels=y)

        self.qp_ = self._q_to_qp(corr_to_target=corr_to_target, corr_features=corr_features, coeff_cap=self.coeff_cap)

        return self

    def transform(self, X):
        check_is_fitted(self)
        return self.qp_

#
# class QuboTransformer(BaseEstimator, TransformerMixin):
#
#     def __init__(self,
#                  alpha: int = 10,
#                  beta: int = 1000,
#                  k: int = 5,
#                  shots: int = 10000):
#         self.alpha = alpha
#         self.beta = beta
#         self.k = k
#         self.shots = shots
#
#     @staticmethod
#     def _data_to_qubo(features: np.ndarray, labels: np.ndarray, alpha: int = 10, beta: int = 1000,
#                       k: int = 5) -> np.array:
#         """
#         Method for transforming input data to QUBO shape
#
#         Args:
#             features: np.array of features
#             labels: np.array of target column
#             alpha: parameter
#             beta: parameter
#             k: number of features to be selected
#
#         Returns: np.array
#
#         """
#         features = features.astype(float)
#         labels = labels.astype(float)
#         B = features.T  # B ... incidence matrix of hypergraph
#         l = labels  # l ... labels
#         n = B.shape[0]  # n ... number of features
#         L = B @ B.T  # L+ ... positive Laplacian matrix, L+ = D + Aw
#         D = np.diag(np.diag(L))  # D ... degree matrix
#         Aw = L - D  # Aw ... weighted adjacency matrix
#         Df = np.diag(B @ l)  # Df ... filtered degree matrix
#         ub_Aw = np.ones(n) @ Aw @ np.ones(n)  # upper bound
#         ub_Df = np.ones(n) @ Df @ np.ones(n)  # upper bound
#         alpha_ub = ub_Aw / ub_Df
#         alpha = alpha_ub * 1.01  # to make sure (xT Aw x) < alpha (xT Df x)
#         Qp = Aw - alpha * Df  # Qp ... "problem"
#         Qc = beta * (np.ones(n) - 2 * k * np.eye(n))  # Qc ... constraints
#         # Qc ... "constraint". This is a cardinality constraint which is minimal when 1@x==k.
#         Q = Qp + Qc
#         return Q
#
#     @staticmethod
#     def _q_to_qp(Q: np.ndarray) -> QuadraticProgram:
#         """
#         This method transforms the input matrix Q into a QuadraticProgram
#
#         Args:
#             Q: input matrix
#
#         Returns: QuadraticProgram
#
#         """
#         n = Q.shape[0]
#         qp_name = 'QuanTeam'
#         linear_terms = {"x" + str(i): x_i for i, x_i in enumerate(np.diag(Q))}
#         quadratic_terms = {("x" + str(i), "x" + str(j)):
#                                2 * Q[i, j]
#                            for i in range(n)
#                            for j in range(n)
#                            if i < j}
#         qp = QuadraticProgram(qp_name)
#         for i in range(n):
#             qp.binary_var(name="x" + str(i))
#
#         qp.minimize(constant=0, linear=linear_terms, quadratic=quadratic_terms)
#
#         return qp
#
#     def fit(self, X, y=None):
#         if y:
#             q = self._data_to_qubo(features=X.values, labels=y, alpha=self.alpha, beta=self.beta, k=self.k)
#         else:
#             q = X.values
#         self.qp_ = self._q_to_qp(q)
#         return self
#
#     def transform(self, X):
#         check_is_fitted(self)
#         return self.qp_
