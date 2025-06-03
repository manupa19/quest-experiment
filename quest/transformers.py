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

    def fit(self, X: np.array): # is learning how to preprocess X.
        """
        Here, fitting means binning numeric data and one-hot-encoding categorical data.

        Args:
            X: array to be transformed
        """
        self.transformer_ = make_column_transformer( #creates a scikit-learn ColumnTransformer — it knows how to process different columns in different ways
                        (KBinsDiscretizer(n_bins=self.quantiles, encode='onehot-dense',  strategy='quantile'),
                         make_column_selector(dtype_include=np.number)),
                        (OneHotEncoder(sparse_output=False,  handle_unknown="infrequent_if_exist"),
                         make_column_selector(dtype_include=[object, 'category']))
        ).fit(X)  #trains that transformer: it learns:
#how to split your numeric columns into quantile bins
#what categories exist in your categorical columns

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

    def __init__(self, coeff_cap: float = 2, max_features: int = 100, scoring_method: str = 'correlation'): 
        self.coeff_cap = coeff_cap
        self.max_features = max_features
        self.scoring_method = scoring_method  # 'iv' or 'correlation'


    @staticmethod
    def calculate_iv(feature: pd.Series, label: pd.Series) -> float:
        df = pd.DataFrame({'feature': feature, 'label': label})
        total_good = (label == 0).sum()
        total_bad = (label == 1).sum()

        iv=0
        eps = 1e-6
        
        for val, group in df.groupby('feature'):
            good = (group['label'] == 0).sum()
            bad = (group['label'] == 1).sum()

            dist_good = good/total_good if total_good > 0 else eps
            dist_bad = bad/total_bad if total_bad > 0 else eps

            if dist_good == 0 or dist_bad == 0:
                continue

            woe = np.log(dist_good / dist_bad)
            iv += (dist_good - dist_bad) * woe

        return iv
            
            
    
    def _data_to_iv(self, features: np.ndarray, labels: np.ndarray) -> (pd.DataFrame, pd.DataFrame):
        if not isinstance(labels, pd.DataFrame):
             data = pd.DataFrame(features)
        else:
            data = features
        data_label = data.copy()
        if isinstance(labels, pd.DataFrame) or isinstance(labels, pd.Series):
            data_label['label'] = labels.values
        else:
            data_label['label'] = labels
            
        iv_scores = {}
        for col in data_label.columns:
            if col == 'label':
                continue
            iv = self.calculate_iv(data_label[col], data_label['label'])
            iv_scores[col] = iv

        iv_series = pd.Series(iv_scores).sort_values(ascending = False).dropna()
        iv_series = iv_series[:self.max_features]

        self.pre_selected_features = iv_series.index
        corr_features = data.loc[:, self.pre_selected_features].corr()
        
        return iv_series, corr_features
            
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
    def _q_to_qp(feature_scores: pd.DataFrame, corr_features : pd.DataFrame, coeff_cap: float = 2) -> QuadraticProgram:
        print("Feature scores:")
        print(feature_scores)
        print("Correlation matrix:")
        print(corr_features)

        print("Number of features:", len(feature_scores))
        print("Quadratic matrix shape:", corr_features.shape)

        # -- Overall Params
        qp_name = 'QUEST'

        # --------------
        var_selected = []
        var_tot = [var for i, var in zip(range(len(feature_scores.index)), feature_scores.index)]

        def feature_score_func(field_i):
            val = feature_scores.loc[field_i]
            return round(val.values[0], 6) if isinstance(val, pd.Series) else round(val, 6)

        def corr_func(field_i, field_j):
            return round(corr_features.loc[field_i, field_j], 6)

        def index_to_field(i):
            return feature_scores.index[i]

        # Define the qp (quadratic problem)
        qp = QuadraticProgram(qp_name)
        for i in range(len(var_tot)):
            qp.binary_var(name="a" + str(i))  # these are the a_i
        linear_terms = {"a" + str(i): abs(feature_score_func(i_field)) for i, i_field in
                        zip(range(len(var_tot)), var_tot)} #abs is not needed for iv but it won't cause any problems and is needed for correlation so it must be here

        # Note that we apply here the abs(...) of the correlation among featuers.
        quadratic_terms = {("a" + str(i), "a" + str(j)):
                               - coeff_cap * abs(corr_func(index_to_field(i), index_to_field(j)))
                           for i in range(len(var_tot))
                           for j in range(len(var_tot))
                           if i < j}

        qp.maximize(constant=0, linear=linear_terms, quadratic=quadratic_terms)

        return qp

    def fit(self, X, y=None):
        if self.scoring_method == 'correlation':
            corr_to_target, corr_features = self._data_to_corr(features=X, labels=y)
            self.qp_ = self._q_to_qp(feature_scores=corr_to_target, corr_features=corr_features, coeff_cap=self.coeff_cap)
        else:
            iv_to_target, corr_features = self._data_to_iv(features=X, labels=y)
            self.qp_ = self._q_to_qp(feature_scores=iv_to_target, corr_features=corr_features, coeff_cap=self.coeff_cap)

        return self

    def transform(self, X):
        check_is_fitted(self)
        return self.qp_
