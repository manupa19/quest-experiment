from typing import Optional, Any

from abc import ABC, abstractmethod

from qiskit_optimization import QuadraticProgram
from qiskit_algorithms.minimum_eigensolvers import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA, SciPyOptimizer
from qiskit_ibm_runtime import Sampler, Session, QiskitRuntimeService, Options
from qiskit_ibm_runtime.accounts import ChannelType
from qiskit_optimization.algorithms import CplexOptimizer
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator

from .utils import logger

_logger = logger(__name__)


class AbstractSolver(ABC):

    def __init__(self, qp):
        self.qp = qp
        pass

    @property
    @abstractmethod
    def service(self):
        pass

    @property
    @abstractmethod
    def circuit(self):
        pass

    @abstractmethod
    def run(self):
        pass


class QAOAQiskitSolver(AbstractSolver):

    def __init__(self,
                 qp: QuadraticProgram,
                 backend: str = 'ibm_kyoto',
                 channel: Optional[ChannelType] = None,
                 token: Optional[str] = None,
                 verify: bool = False,
                 instance: str = 'erste-digital/quest/general',
                 optimizer: SciPyOptimizer = COBYLA(),
                 passmanager_options: dict = None,
                 sampler_options: dict[dict] = None):
        super().__init__(qp)
        if passmanager_options is None:
            passmanager_options = {"optimization_level": 3}
        if sampler_options is None:
            sampler_options = {}
        self.backend = backend
        self.channel = channel
        self.token = token
        self.verify = verify
        self.instance = instance
        self.optimizer = optimizer
        self.sampler_options = sampler_options
        self.passmanager_options = passmanager_options
        self.hamiltonian, self.offset = self.qp.to_ising()

    @property
    def service(self):
        service = QiskitRuntimeService(verify=self.verify, token=self.token, channel=self.channel,
                                       instance=self.instance)
        return service

    @property
    def circuit(self):
        return QAOAAnsatz(self.hamiltonian, reps=1, initial_state=None, mixer_operator=None)

    def isa_circuit(self):
        target = self.service.get_backend(name=self.backend).target
        pm = generate_preset_pass_manager(target=target, **self.passmanager_options)
        ansatz_isa = pm.run(self.circuit)
        isa_hamiltonian = self.hamiltonian.apply_layout(ansatz_isa.layout)
        return ansatz_isa, isa_hamiltonian

    # @property
    # def circuit_width(self):
    #     return self.isa_circuit.width()
    #
    # @property
    # def circuit_depth(self):
    #     return self.isa_circuit.decompose().depth()
    #
    # @property
    # def circuit_gates(self):
    #     return self.isa_circuit.count_ops()

    # @property
    # def isa_hamiltonian(self):
    #     return

    def run(self):
        options = Options(**self.sampler_options)
        options.transpilation.skip_transpilation = True
        with Session(service=self.service, #within that session (with the IBM Quantum runtime environment)
                     backend=self.backend):
            # _logger.info(f"Number of Gates: {self.circuit_gates}")
            # _logger.info(f"Depth of Circuit: {self.circuit_depth}")
            # _logger.info(f"Number of Qubits of Backend: {self.circuit_width}")
            _logger.info(f"Backend: {self.backend}")

            sampler = Sampler(options=options) #a class from Qiskit Runtime that is used to run quantum circuits and return measurement probabilities, basically a backend runner
            self.ansatz_isa, isa_hamiltonian = self.isa_circuit()
            svqe = SamplingVQE(sampler, self.ansatz_isa, optimizer=self.optimizer)
            result = svqe.compute_minimum_eigenvalue(isa_hamiltonian)

            result.best_measurement['value'] = result.best_measurement['value'] + self.offset

            return result


class CplexSolver(AbstractSolver):

    def __init__(self, qp: QuadraticProgram):
        super().__init__(qp)

    def service(self):
        pass

    def circuit(self):
        pass
        
    def run(self):
        print("Starting CPLEX solve...", flush=True)
        cplex = CplexOptimizer()

        result = cplex.solve(self.qp)
        print("CPLEX solve finished.", flush=True)
        return result

    # def run(self):
    #     cplex = CplexOptimizer()
    #     return cplex.solve(self.qp)


class AerSolver(QAOAQiskitSolver):

    def __init__(self, aersim_params: dict = None, **kwargs):
        super().__init__(**kwargs)
        if aersim_params is None:
            aersim_params = {}
        self.aersim_params = aersim_params
        self.sim = AerSimulator(**self.aersim_params)

    def isa_circuit(self):
        pm = generate_preset_pass_manager(backend=self.sim, **self.passmanager_options)
        ansatz_isa = pm.run(self.circuit)
        isa_hamiltonian = self.hamiltonian.apply_layout(ansatz_isa.layout)
        return ansatz_isa, isa_hamiltonian

    def run(self):
        options = Options(**self.sampler_options)
        options.transpilation.skip_transpilation = True
        with Session(
                     backend=self.sim) as session:
            isa_circuit, isa_hamiltonian = self.isa_circuit()
            sampler = Sampler(session=session, options=options)
            svqe = SamplingVQE(sampler, isa_circuit, optimizer=self.optimizer)
            result = svqe.compute_minimum_eigenvalue(isa_hamiltonian)

            result.best_measurement['value'] = result.best_measurement['value'] + self.offset
            return result
