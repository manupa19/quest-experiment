from typing import Optional, Any

from abc import ABC, abstractmethod

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from qiskit_algorithms.optimizers import COBYLA, SciPyOptimizer
from qiskit_ibm_runtime.accounts import ChannelType
from qiskit_optimization import QuadraticProgram

from .transformers import QuboTransformer
from .sdk_integration import QAOAQiskitSolver, CplexSolver, AerSolver
from .utils import logger, unmap_bitstring

_logger = logger(__name__)


class BaseQuest(ABC, BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    @property
    @abstractmethod
    def get_selected_features(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        return self

    @abstractmethod
    def transform(self, X) -> pd.DataFrame:
        return X


class QUEST(BaseQuest):
    """
    QUEST turns feature selection into a QUBO problem, that can be solved by a Quantum Computer.
    """

    def __init__(self,
                 backend: str = 'ibm_kyoto',
                 channel: Optional[ChannelType] = None,
                 token: Optional[str] = None,
                 verify: bool = False,
                 instance: str = 'erste-digital/quest/general',
                 optimizer: SciPyOptimizer = COBYLA(),
                 passmanager_options: dict = None,
                 sampler_options: dict[dict] = None,
                 alpha: int = 10,
                 beta: int = 1000,
                 k: int = 5,
                 shots: int = 10000,
                 coeff_cap: float = 0.5,
                 max_features: int = 100
                 ):
        """

        Args:
            backend: Instance of selected backend, by default the backend is 'ibm_kyoto' on the IBM cloud;
                for running on a quantum computer, input the name of the respective system,
                input string "cplex" for running a classical quantum inspired QUEST
            channel: Channel type. ``ibm_cloud`` or ``ibm_quantum``
            token: IBM Cloud API key or IBM Quantum API token.
            verify: Whether to verify the server's TLS certificate.
            instance:he service instance to use.
                For ``ibm_cloud`` runtime, this is the Cloud Resource Name (CRN) or the service name.
                For ``ibm_quantum`` runtime, this is the hub/group/project in that format.
            optimizer: Specify the optimizer function
            passmanager_options: Parameters of the Passmanager Object
            sampler_options: Parameters of the Sampler primitive
            alpha: Parameter used for defining the QUBO Problem
            beta: Parameter used for defining the QUBO Problem
            k: influences the number fo features to be selected
            shots: Number of repetitions of each circuit, for sampling. If None, the shots are
                extracted from the backend. If the backend has none set, the default is 1024.
        """
        super().__init__()

        self.backend = backend
        self.channel = channel
        self.token = token
        self.verify = verify
        self.instance = instance
        self.optimizer = optimizer
        self.sampler_options = sampler_options
        self.passmanager_options = passmanager_options

        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.shots = shots

        self.max_features = max_features
        self.coeff_cap = coeff_cap

        self.algorithm = QuboTransformer(max_features=max_features, coeff_cap=coeff_cap)

    @staticmethod
    def get_solver(qp: QuadraticProgram):
        """
        Returns the Solver used; gives access to the Solver's methods
        Args:
            qp: The QuadraticProgram to be solved

        Returns:
            QAOAQiskitSolver

        """
        return QAOAQiskitSolver(qp=qp)

    @property
    def get_selected_features(self) -> list:
        """
        Returns the finally selected features after the Qubo feature selection

        Returns: list of selected features

        """
        return self.feature_list_

    @property
    def get_results(self) -> list:
        """
        Returns the finally selected features after the Qubo feature selection

        Returns: list of selected features

        """
        return self.result

    def fit(self, X, y=None):

        qp = self.algorithm.fit_transform(X, y)
        solver = self.get_solver(qp)
        self.result = solver.run()
        _logger.info(f"Results: {self.result}")
        bitstring = unmap_bitstring(self.result.best_measurement['bitstring'], solver.ansatz_isa)
        self.feature_list_ = [X.columns[i] for i, e in enumerate(",".join(bitstring).split(',')) if e == '1']

        _logger.info(f"Results best measurement: {self.result.best_measurement}")

        _logger.info(f"Selected Features: {self.get_selected_features}")

        return self

    def transform(self, X) -> pd.DataFrame:
        _logger.info("Transform Data ...")
        check_is_fitted(self)
        X = X[self.get_selected_features]
        return X


class QUESTInspired(QUEST):

    def __init__(self,
                 max_features: int = 100,
                 coeff_cap: float = 0.5
                 ):
        super().__init__(
                         max_features=max_features,
                         coeff_cap=coeff_cap)


    @staticmethod
    def get_solver(qp: QuadraticProgram):
        """
        Returns the Solver used; gives access to the Solver's methods
        Args:
            qp: The QuadraticProgram to be solved

        Returns:
            CplexSolver

        """
        return CplexSolver(qp=qp)

    def fit(self, X, y=None):
        algo = self.algorithm
        qp = algo.fit_transform(X, y)

        self.result = self.get_solver(qp).run()
        self.feature_list_ = [X.loc[:, algo.pre_selected_features].columns[i] for i, e
                              in enumerate(self.result.variables_dict.values()) if e == 1.0]

        _logger.info(f"Results: {self.result}")

        _logger.info(f"Selected Features: {self.get_selected_features}")

        return self


class QUESTSimulation(QUEST):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_solver(qp: QuadraticProgram):
        """
        Returns the Solver used; gives access to the Solver's methods
        Args:
            qp: The QuadraticProgram to be solved

        Returns:
            AerSolver

        """
        return AerSolver(qp=qp)

    def fit(self, X, y=None):

        qp = self.algorithm.fit_transform(X, y)

        self.result = self.get_solver(qp).run()
        self.feature_list_ = [X.columns[i] for i, e
                              in enumerate(",".join(reversed(self.result.best_measurement['bitstring'])).split(',')) if e == '1']

        _logger.info(f"Results: {self.result.best_measurement}")

        _logger.info(f"Selected Features: {self.get_selected_features}")

        return self



