import numpy as np
import json
import signal
from qiskit.opflow import CircuitSampler, StateFn
from qiskit.circuit import ParameterVector
from qiskit.opflow.evolutions import PauliTrotterEvolution
from qiskit.opflow.expectations import PauliExpectation, AerPauliExpectation
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from exact_simulator import simulation, inner, prep_ham, is_real
from pauli_function import *
from qiskit.utils import QuantumInstance
from qiskit import Aer


# This class aims to simulate the dynamics of a quantum system
# approximating it with a variational ansatz whose parameters
# are varied in order to follow the unitary evolution

# Useful functions
interrupted = False


def signal_handler(signum, frame):
    global interrupted
    print("SIGNAL RECIEVED")
    interrupted = True
    raise KeyboardInterrupt


signal.signal(signal.SIGTERM, signal_handler)


def projector_zero(n_qubits):
    # This function create the global projector |00...0><00...0|
    from qiskit.opflow import Z, I

    prj_list = [0.5 * (I + Z) for i in range(n_qubits)]
    prj = prj_list[0]

    for a in range(1, len(prj_list)):
        prj = prj ^ prj_list[a]

    return prj


def projector_zero_local(n_qubits):
    # This function creates the local version of the cost function
    # proposed by Cerezo et al: https://www.nature.com/articles/s41467-021-21728-w
    from qiskit.opflow import Z, I

    tot_prj = 0

    for k in range(n_qubits):
        prj_list = [I for i in range(n_qubits)]
        prj_list[k] = 0.5 * (I + Z)
        prj = prj_list[0]

        for a in range(1, len(prj_list)):
            prj = prj ^ prj_list[a]

        # print(prj)

        tot_prj += prj

    tot_prj = (1 / n_qubits) * tot_prj

    return tot_prj


def ei(i, n):
    vi = np.zeros(n)
    vi[i] = 1.0
    return vi[:]


class pVQD:
    def __init__(
        self,
        hamiltonian,
        ansatz,
        ansatz_reps,
        parameters,
        initial_shift,
        instance,
        shots,
    ):

        self.hamiltonian = hamiltonian
        self.instance = instance
        self.parameters = parameters
        self.num_parameters = len(parameters)
        self.shift = initial_shift
        self.shots = shots
        self.depth = ansatz_reps
        self.num_qubits = hamiltonian().num_qubits
        print("qubits: ", self.num_qubits)

        ## Initialize quantities that will be equal all over the calculation
        self.params_vec = ParameterVector("p", self.num_parameters)
        # print(ansatz(self.num_qubits, self.depth,None).num_parameters)
        self.base_ansatz = ansatz
        self.ansatz = ansatz(self.num_qubits, self.depth, self.params_vec)

        # ParameterVector for left and right circuit

        self.left = ParameterVector("l", self.ansatz.num_parameters)
        self.right = ParameterVector("r", self.ansatz.num_parameters)

        # ParameterVector for measuring abservables
        self.obs_params = ParameterVector("θ", self.ansatz.num_parameters)

    def construct_total_circuit(
        self, time_step, cost_fun="local", trotter_steps=1, kwargs={}
    ):
        ## This function creates the circuit that will be used to evaluate overlap and its gradient

        # First, create the Trotter step

        step_h = time_step * self.hamiltonian(**kwargs)
        trotter = PauliTrotterEvolution(reps=trotter_steps) 
        U_dt = trotter.convert(step_h.exp_i()).to_circuit()

        l_circ = self.ansatz.assign_parameters({self.params_vec: self.left})
        r_circ = self.ansatz.assign_parameters({self.params_vec: self.right})

        ## Projector
        if cost_fun == "local":
            zero_prj = StateFn(
                projector_zero_local(self.num_qubits), is_measurement=True
            )
        elif cost_fun == "gloabl":
            zero_prj = StateFn(projector_zero(self.num_qubits), is_measurement=True)
        state_wfn = zero_prj @ StateFn(r_circ + U_dt + l_circ.inverse())

        return state_wfn

    def construct_initial_circuit(self, cost_fun="local", statevector=None):
        ## This function creates the circuit that will be used to evaluate overlap and its gradient

        l_circ = QuantumCircuit(self.num_qubits)
        l_circ.append(StatePreparation(statevector), list(range(self.num_qubits)))
        r_circ = self.ansatz.assign_parameters({self.params_vec: self.left})

        ## Projector
        if cost_fun == "local":
            zero_prj = StateFn(
                projector_zero_local(self.num_qubits), is_measurement=True
            )
        elif cost_fun == "gloabl":
            zero_prj = StateFn(projector_zero(self.num_qubits), is_measurement=True)
        state_wfn = zero_prj @ StateFn(r_circ + l_circ.inverse())
        return state_wfn

    # This function calculate overlap and gradient of the overlap

    def compute_overlap_and_gradient(
        self, state_wfn, parameters, shift, expectator, sampler
    ):

        nparameters = len(parameters)
        # build dictionary of parameters to values
        # {left[0]: parameters[0], .. ., right[0]: parameters[0] + shift[0], ...}

        # First create the dictionary for overlap
        values_dict = [
            dict(
                zip(
                    self.right[:] + self.left[:],
                    parameters.tolist() + (parameters + shift).tolist(),
                )
            )
        ]

        # Then the values for the gradient
        for i in range(nparameters):
            values_dict.append(
                dict(
                    zip(
                        self.right[:] + self.left[:],
                        parameters.tolist()
                        + (
                            parameters + shift + ei(i, nparameters) * np.pi / 2.0
                        ).tolist(),
                    )
                )
            )
            values_dict.append(
                dict(
                    zip(
                        self.right[:] + self.left[:],
                        parameters.tolist()
                        + (
                            parameters + shift - ei(i, nparameters) * np.pi / 2.0
                        ).tolist(),
                    )
                )
            )

        # Now evaluate the circuits with the parameters assigned
        results = []

        for values in values_dict:
            sampled_op = sampler.convert(state_wfn, params=values)

            mean = sampled_op.eval().real
            # mean  = np.power(np.absolute(mean),2)
            est_err = 0

            if not self.instance.is_statevector:
                variance = expectator.compute_variance(sampled_op).real
                est_err = np.sqrt(variance / self.shots)

            results.append([mean, est_err])

        E = np.zeros(2)
        g = np.zeros((nparameters, 2))

        E[0], E[1] = results[0]

        for i in range(nparameters):
            rplus = results[1 + 2 * i]
            rminus = results[2 + 2 * i]
            # G      = (Ep - Em)/2
            # var(G) = var(Ep) * (dG/dEp)**2 + var(Em) * (dG/dEm)**2
            g[i, :] = (rplus[0] - rminus[0]) / 2.0, np.sqrt(
                rplus[1] ** 2 + rminus[1] ** 2
            ) / 2.0

        self.overlap = E
        self.gradient = g

        return E, g

    ## this function does the same thing but uses SPSA

    def compute_overlap_and_gradient_spsa(
        self, state_wfn, parameters, shift, expectator, sampler, count
    ):

        nparameters = len(parameters)
        # build dictionary of parameters to values
        # {left[0]: parameters[0], .. ., right[0]: parameters[0] + shift[0], ...}

        # Define hyperparameters
        c = 0.1
        a = 0.16
        A = 1
        alpha = 0.602
        gamma = 0.101

        a_k = a / np.power(A + count, alpha)
        c_k = c / np.power(count, gamma)

        # Determine the random shift

        delta = np.random.binomial(1, 0.5, size=nparameters)
        delta = np.where(delta == 0, -1, delta)
        delta = c_k * delta

        # First create the dictionary for overlap
        values_dict = [
            dict(
                zip(
                    self.right[:] + self.left[:],
                    parameters.tolist() + (parameters + shift).tolist(),
                )
            )
        ]

        # Then the values for the gradient

        values_dict.append(
            dict(
                zip(
                    self.right[:] + self.left[:],
                    parameters.tolist() + (parameters + shift + delta).tolist(),
                )
            )
        )
        values_dict.append(
            dict(
                zip(
                    self.right[:] + self.left[:],
                    parameters.tolist() + (parameters + shift - delta).tolist(),
                )
            )
        )

        # Now evaluate the circuits with the parameters assigned

        results = []

        for values in values_dict:
            sampled_op = sampler.convert(state_wfn, params=values)

            mean = sampled_op.eval()[0]
            mean = np.power(np.absolute(mean), 2)
            est_err = 0

            if not self.instance.is_statevector:
                variance = expectator.compute_variance(sampled_op)[0].real
                est_err = np.sqrt(variance / self.shots)

            results.append([mean, est_err])

        E = np.zeros(2)
        g = np.zeros((nparameters, 2))

        E[0], E[1] = results[0]

        # and the gradient

        rplus = results[1]
        rminus = results[2]

        for i in range(nparameters):
            # G      = (Ep - Em)/2Δ_i
            # var(G) = var(Ep) * (dG/dEp)**2 + var(Em) * (dG/dEm)**2
            g[i, :] = a_k * (rplus[0] - rminus[0]) / (2.0 * delta[i]), np.sqrt(
                rplus[1] ** 2 + rminus[1] ** 2
            ) / (2.0 * delta[i])

        self.overlap = E
        self.gradient = g

        return E, g

    def measure_aux_ops(self, obs_wfn, pauli, parameters, expectator, sampler):

        # This function calculates the expectation value of a given operator

        # Prepare the operator and the parameters
        wfn = StateFn(obs_wfn)
        op = StateFn(pauli, is_measurement=True)
        values_obs = dict(zip(self.obs_params[:], parameters.tolist()))

        braket = op @ wfn

        grouped = expectator.convert(braket)
        sampled_op = sampler.convert(grouped, params=values_obs)

        # print(sampled_op.eval())
        mean_value = sampled_op.eval().real
        est_err = 0

        if not self.instance.is_statevector:
            variance = expectator.compute_variance(sampled_op).real
            est_err = np.sqrt(variance / self.shots)

        res = [mean_value, est_err]

        return res

    def adam_gradient(self, count, m, v, g):
        ## This function implements adam optimizer
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        alpha = [0.001 for i in range(len(self.parameters))]
        if count == 0:
            count = 1

        new_shift = [0 for i in range(len(self.parameters))]

        for i in range(len(self.parameters)):
            m[i] = beta1 * m[i] + (1 - beta1) * g[i]
            v[i] = beta2 * v[i] + (1 - beta2) * np.power(g[i], 2)

            alpha[i] = (
                alpha[i]
                * np.sqrt(1 - np.power(beta2, count))
                / (1 - np.power(beta1, count))
            )

            new_shift[i] = self.shift[i] + alpha[i] * (m[i] / (np.sqrt(v[i]) + eps))

        return new_shift

    def prepare_arb_state(
        self,
        statevector,
        initial_point,
        cost_fun="local",
        opt="sgd",
        grad="param_shift",
        ths=0.99999,
        max_iter=60,
    ):
        if np.any(initial_point) != None:
            if len(initial_point) != len(self.parameters):
                print(
                    "TypeError: Initial parameters are not of the same size of circuit parameters"
                )
                return

            self.parameters = initial_point

        print("preparing arbitrary state")
        state_wfn = self.construct_initial_circuit(cost_fun, statevector)
        if self.instance.is_statevector and cost_fun == "local":
            expectation = AerPauliExpectation()
        else:
            expectation = PauliExpectation()

        sampler = CircuitSampler(self.instance)

        state_wfn = expectation.convert(state_wfn)

        count = 0
        self.overlap = [0.01, 0]

        if opt == "adam":
            m = np.zeros(len(self.parameters))
            v = np.zeros(len(self.parameters))

        if opt == "momentum":
            old_grad = np.zeros(len(self.parameters))
            g = np.zeros((len(self.parameters), 2))

        while self.overlap[0] < ths and count < max_iter:
            # print("Shift optimizing step:", count + 1)
            count = count + 1

            if opt == "momentum":
                old_grad = np.asarray(g[:, 0])
            ## Measure energy and gradient

            if grad == "param_shift":
                E, g = self.compute_overlap_and_gradient(
                    state_wfn, self.parameters, self.shift, expectation, sampler
                )
            if grad == "spsa":
                E, g = self.compute_overlap_and_gradient_spsa(
                    state_wfn,
                    self.parameters,
                    self.shift,
                    expectation,
                    sampler,
                    count,
                )

            if count == 1:
                initial_fidelities = self.overlap[0]
                err_init_fid = self.overlap[1]

            if count % 50 == 0:
                print(self.overlap[0])

            if opt == "adam":
                print("\n Adam \n")
                meas_grad = np.asarray(g[:, 0])
                self.shift = np.asarray(self.adam_gradient(count, m, v, meas_grad))

            if opt == "momentum":
                print("Momentum")
                m_grad = np.asarray(g[:, 0]) + 0.9 * old_grad
                self.shift = self.shift + m_grad

            elif opt == "sgd":
                self.shift = self.shift + g[:, 0]
            # Norm of the gradient
            g_vec = np.asarray(g[:, 0])
            g_norm = np.linalg.norm(g_vec)

        print(f"initial fidelity: {initial_fidelities} +- {err_init_fid}")
        print(f"final fidelity: {self.overlap[0]} +- {self.overlap[1]}")

        return self.parameters + self.shift

    def run(
        self,
        ths,
        dt,
        n_steps,
        obs_dict={},
        filename="algo_result.dat",
        max_iter=100,
        opt="sgd",
        cost_fun="global",
        grad="param_shift",
        initial_point=None,
        initial_parameters={
            "position": -9,
            "velocity": 0,
            "trotter": 1,
            "parameterized": True,
            "padding": 10,
        },
        trotter_steps=1,
    ):
        log_data = {}
        log_data["QC_energy_el"] = []
        log_data["dt"] = initial_parameters["dt"]
        log_data["depth"] = self.depth
        log_data["r_r"] = initial_parameters["r_r"]
        log_data["r_f"] = initial_parameters["r_f"]
        log_data["r_l"] = initial_parameters["r_l"]
        log_data["L"] = initial_parameters["L"]
        log_data["res"] = initial_parameters["res"]
        log_data["trotter_steps"] = trotter_steps
        log_data["parameterized"] = initial_parameters["parameterized"]
        log_data["padding"] = initial_parameters["padding"]
        log_data["fidelity_to_ideal"] = []
        log_data["fidelity_to_exact"] = [1]
        log_data["QC_energy_Tnuc"] = []
        log_data["QC_energy_Vnuc"] = []
        log_data["QC_coefficients"] = []
        log_data["QC_energy"] = [] 
        log_data['ideal_energy'] = []
        log_data['total_energy'] = []
        
        print("running exact computation")

        ideal = simulation(
            dt=initial_parameters["dt"],
            steps=n_steps,
            r_f=initial_parameters["r_f"],
            r_l=initial_parameters["r_l"],
            r_r=initial_parameters["r_r"],
            resolution=initial_parameters["res"],
            L=initial_parameters["L"],
            x_0=initial_parameters["position"],
            v_0=initial_parameters["velocity"],
            vector=initial_parameters["vector"],
            # parameterized=initial_parameters["parameterized"],
            padding=initial_parameters["padding"],
        )

        backend_statevector = Aer.get_backend("statevector_simulator")
        self.instance_statevector = QuantumInstance(backend=backend_statevector)

        qc_WF = np.asarray(
            self.instance_statevector.backend.run(
                self.base_ansatz(
                    np.int(np.log2(log_data["res"])), log_data["depth"], initial_point
                )
            )
            .result()
            .get_statevector()
        )
        
        log_data["fidelity_to_ideal"].append(
            np.abs(np.inner(np.conjugate(qc_WF), ideal.store_psi[0])) ** 2
        )

        exact = simulation(
            dt=initial_parameters["dt"],
            steps=n_steps,
            r_f=initial_parameters["r_f"],
            r_l=initial_parameters["r_l"],
            r_r=initial_parameters["r_r"],
            resolution=initial_parameters["res"],
            L=initial_parameters["L"],
            x_0=initial_parameters["position"],
            v_0=initial_parameters["velocity"],
            vector=qc_WF,
            # parameterized=initial_parameters["parameterized"],
            padding=initial_parameters["padding"],
        )

        H0 = prep_ham(
            r_nuc = initial_parameters["position"],
            r_f=initial_parameters["r_f"],
            r_l=initial_parameters["r_l"],
            r_r=initial_parameters["r_r"],
            resolution=initial_parameters["res"],
            L=initial_parameters["L"],
            padding=initial_parameters["padding"],
        )
        
        log_data["QC_energy_el"].append(is_real(inner(qc_WF, H0[0])))
        log_data["QC_energy_Tnuc"].append(initial_parameters["velocity"] * initial_parameters["velocity"] * 0.5 * 1836)
        log_data["QC_energy_Vnuc"].append(H0[6][0])
        log_data["QC_energy"].append(log_data["QC_energy_el"][-1]+log_data["QC_energy_Tnuc"][-1]+log_data["QC_energy_Vnuc"][-1])
        log_data["QC_coefficients"].append((np.abs(np.conj(H0[2]).T @ qc_WF) ** 2).tolist())


        # log_data["QC_forces"].append(is_real(inner(qc_WF,H0[3])))

        ### Ideal results
        log_data['ideal_coefficients'] = ideal.store_c.tolist()
        log_data["ideal_forces_el"] = ideal.store_fel.tolist()
        log_data["ideal_forces_nuc"] = ideal.store_fnuc.tolist()
        log_data["ideal_tot_forces"] = ideal.store_f.tolist()
        log_data["ideal_velocities"] = ideal.store_v.tolist()
        log_data["ideal_positions"] = ideal.store_xs.tolist()
        log_data["ideal_energy_el"] = ideal.store_energy_el.tolist()
        log_data["ideal_energy_Tnuc"] = ideal.store_energy_Tnuc.tolist()
        log_data["ideal_energy_Vnuc"] = ideal.store_energy_Vnuc.tolist()
        log_data["ideal_energy"] = (ideal.store_energy_el+ideal.store_energy_Tnuc+ideal.store_energy_Vnuc).tolist()
        ### Exact results
        log_data['exact_coefficients'] = exact.store_c.tolist()
        log_data["exact_forces_el"] = exact.store_fel.tolist()
        log_data["exact_forces_nuc"] = exact.store_fnuc.tolist()
        log_data["exact_tot_forces"] = exact.store_f.tolist()
        log_data["exact_velocities"] = exact.store_v.tolist()
        log_data["exact_positions"] = exact.store_xs.tolist()
        log_data["exact_energy_el"] = exact.store_energy_el.tolist()
        log_data["exact_energy_Tnuc"] = exact.store_energy_Tnuc.tolist()
        log_data["exact_energy_Vnuc"] = exact.store_energy_Vnuc.tolist()
        log_data["exact_energy"] = (exact.store_energy_el+exact.store_energy_Tnuc+exact.store_energy_Vnuc).tolist()

        ## initialize useful quantities once
        if self.instance.is_statevector and cost_fun == "local":
            expectation = AerPauliExpectation()
        else:
            expectation = PauliExpectation()

        sampler = CircuitSampler(self.instance)

        ## Now prepare the state in order to compute the overlap and its gradient
        state_wfn = self.construct_total_circuit(
            dt,
            cost_fun,
            trotter_steps,
            {
                "r_nuc": initial_parameters["position"],
                "r_f": initial_parameters["r_f"],
                "r_l": initial_parameters["r_l"],
                "r_r": initial_parameters["r_r"],
                "resolution": initial_parameters["res"],
                "L": initial_parameters["L"],
                "padding": initial_parameters["padding"],
            },
        )

        state_wfn = expectation.convert(state_wfn)

        ## Also the standard state for measuring the observables
        obs_wfn = self.ansatz.assign_parameters({self.params_vec: self.obs_params})

        #######################################################

        times = []
        tot_steps = 0

        if np.any(initial_point) != None:
            if len(initial_point) != len(self.parameters):
                print(
                    "TypeError: Initial parameters are not of the same size of circuit parameters"
                )
                return

            print("\nRestart from: ")
            print(initial_point)
            self.parameters = initial_point

        print("Running the algorithm")

        # Measure force
        if len(obs_dict) > 0:
            obs_measure = {}
            obs_error = {}

            for (obs_name, obs_pauli) in obs_dict.items():
                first_measure = self.measure_aux_ops(
                    obs_wfn,
                    obs_pauli(
                        r_nuc=initial_parameters["position"],
                        r_f=initial_parameters["r_f"],
                        r_l=initial_parameters["r_l"],
                        r_r=initial_parameters["r_r"],
                        resolution=initial_parameters["res"],
                        L=initial_parameters["L"],
                        padding=initial_parameters["padding"],
                    ),
                    self.parameters,
                    expectation,
                    sampler,
                )

                obs_measure[str(obs_name)] = [first_measure[0]]
                obs_error["err_" + str(obs_name)] = [first_measure[1]]

        counter = []
        initial_fidelities = []
        fidelities = []
        err_fin_fid = []
        err_init_fid = []
        forces = []
        forces_nuc = []
        forces_el = []
        params = []
        positions = [initial_parameters["position"]]  # x_0
        velocities = [initial_parameters["velocity"]]  # v_0
        force_nuc = (
            -initial_parameters["nuc_force"](
                positions[0],
                r_f=initial_parameters["r_f"],  # f_0el
                r_l=initial_parameters["r_l"],
                r_r=initial_parameters["r_r"],
                resolution=initial_parameters["res"],
                L=initial_parameters["L"],
                padding=initial_parameters["padding"],
            )
            / 1836
        )[0]
        force_el = -obs_measure["force"][0] / 1836  # f_0nuc
        if initial_parameters["parameterized"]:
            force = log_data["ideal_tot_forces"][0]
        else:
            force = force_nuc + force_el  # f_0
        forces_el.append(force_el)
        forces_nuc.append(force_nuc)
        forces.append(force)
        positions.append(
            positions[0] + velocities[0] * dt + 0.5 * (forces[0]) * dt * dt
        )  # x_1

        params.append(list(self.parameters))
        i = 0
        ###################
        # compute f1
        state_wfn = self.construct_total_circuit(
            dt,
            cost_fun,
            trotter_steps,
            {
                "r_nuc": positions[-1],
                "r_f": initial_parameters["r_f"],
                "r_l": initial_parameters["r_l"],
                "r_r": initial_parameters["r_r"],
                "resolution": initial_parameters["res"],
                "L": initial_parameters["L"],
                "padding": initial_parameters["padding"],
            },
        )
        state_wfn = expectation.convert(state_wfn)

        ##  state for measuring the observables
        obs_wfn = self.ansatz.assign_parameters({self.params_vec: self.obs_params})

        count = 0
        self.overlap = [0.01, 0]
        g_norm = 1

        if opt == "adam":
            m = np.zeros(len(self.parameters))
            v = np.zeros(len(self.parameters))

        if opt == "momentum":
            old_grad = np.zeros(len(self.parameters))
            g = np.zeros((len(self.parameters), 2))

        while self.overlap[0] < ths and count < max_iter:
            # print("Shift optimizing step:", count + 1)
            count = count + 1

            if opt == "momentum":
                old_grad = np.asarray(g[:, 0])
            ## Measure energy and gradient

            if grad == "param_shift":
                E, g = self.compute_overlap_and_gradient(
                    state_wfn, self.parameters, self.shift, expectation, sampler
                )
            if grad == "spsa":
                E, g = self.compute_overlap_and_gradient_spsa(
                    state_wfn,
                    self.parameters,
                    self.shift,
                    expectation,
                    sampler,
                    count,
                )

            tot_steps = tot_steps + 1

            if count == 1:
                initial_fidelities.append(self.overlap[0])
                err_init_fid.append(self.overlap[1])

            # print("Overlap", self.overlap)
            # print("Gradient", self.gradient[:, 0])

            if opt == "adam":
                print("\n Adam \n")
                meas_grad = np.asarray(g[:, 0])
                self.shift = np.asarray(self.adam_gradient(count, m, v, meas_grad))

            if opt == "momentum":
                print("Momentum")
                m_grad = np.asarray(g[:, 0]) + 0.9 * old_grad
                self.shift = self.shift + m_grad

            elif opt == "sgd":
                self.shift = self.shift + g[:, 0]

            # Norm of the gradient
            g_vec = np.asarray(g[:, 0])
            g_norm = np.linalg.norm(g_vec)

        # Update parameters

        # print("\n---------------------------------- \n")

        # print("Shift after optimizing:", self.shift)
        # print("New parameters:", self.parameters + self.shift)
        # print("New overlap: ", self.overlap[0])

        self.parameters = self.parameters + self.shift

        # Measure quantities and save them

        if len(obs_dict) > 0:

            for (obs_name, obs_pauli) in obs_dict.items():
                run_measure = self.measure_aux_ops(
                    obs_wfn,
                    obs_pauli(
                        r_nuc=positions[-1],
                        r_f=initial_parameters["r_f"],
                        r_l=initial_parameters["r_l"],
                        r_r=initial_parameters["r_r"],
                        resolution=initial_parameters["res"],
                        L=initial_parameters["L"],
                        padding=initial_parameters["padding"],
                    ),
                    self.parameters,
                    expectation,
                    sampler,
                )
                obs_measure[str(obs_name)].append(run_measure[0])
                obs_error["err_" + str(obs_name)].append(run_measure[1])

        counter.append(count)
        fidelities.append(self.overlap[0])
        err_fin_fid.append(self.overlap[1])

        force_el = -obs_measure["force"][-1] / 1836  # f_1el
        force_nuc = (
            -initial_parameters["nuc_force"](
                positions[-1],
                r_f=initial_parameters["r_f"],  # f1_nuc
                r_l=initial_parameters["r_l"],
                r_r=initial_parameters["r_r"],
                resolution=initial_parameters["res"],
                L=initial_parameters["L"],
                padding=initial_parameters["padding"],
            )
            / 1836
        )[0]

        if initial_parameters["parameterized"]:
            force = log_data["ideal_tot_forces"][1]
        else:
            force = force_nuc + force_el  # f1

        qc_WF = np.asarray(
            self.instance_statevector.backend.run(
                self.base_ansatz(
                    np.int(np.log2(log_data["res"])), log_data["depth"], self.parameters
                )
            )
            .result()
            .get_statevector()
        )

        H0 = prep_ham(
            positions[-1],
            r_f=initial_parameters["r_f"],
            r_l=initial_parameters["r_l"],
            r_r=initial_parameters["r_r"],
            resolution=initial_parameters["res"],
            L=initial_parameters["L"],
            padding=initial_parameters["padding"],
        )

        log_data["QC_coefficients"].append((np.abs(np.conj(H0[2]).T @ qc_WF) ** 2).tolist())

        log_data["fidelity_to_ideal"].append(
            np.abs(np.inner(np.conjugate(qc_WF), ideal.store_psi[1])) ** 2
        )
        log_data["fidelity_to_exact"].append(
            np.abs(np.inner(np.conjugate(qc_WF), exact.store_psi[1])) ** 2
        )
        
        
        forces_el.append(force_el)
        forces_nuc.append(force_nuc)
        forces.append(force)
        velocities.append(velocities[0] + dt * ((forces[0] + forces[1]) / 2))  # v_1
        log_data["QC_energy_el"].append(is_real(inner(qc_WF, H0[0])))
        log_data["QC_energy_Tnuc"].append(velocities[-1] * velocities[-1] * 0.5 * 1836)
        log_data["QC_energy_Vnuc"].append(H0[6][0])
        log_data["QC_energy"].append(log_data["QC_energy_el"][-1]+log_data["QC_energy_Tnuc"][-1]+log_data["QC_energy_Vnuc"][-1])

        try:
            while i < n_steps and not interrupted:

                print("Time slice:", i + 1)
                positions.append(
                    positions[-1] + velocities[-1] * dt + 0.5 * ((forces[-1])) * (dt**2)
                )

                params.append(list(self.parameters))
                times.append(i * dt)
                i += 1

                state_wfn = self.construct_total_circuit(
                    dt,
                    cost_fun,
                    trotter_steps,
                    {
                        "r_nuc": positions[-1],
                        "r_f": initial_parameters["r_f"],
                        "r_l": initial_parameters["r_l"],
                        "r_r": initial_parameters["r_r"],
                        "resolution": initial_parameters["res"],
                        "L": initial_parameters["L"],
                        "padding": initial_parameters["padding"],
                    },
                )
                state_wfn = expectation.convert(state_wfn)

                ## Also the standard state for measuring the observables
                obs_wfn = self.ansatz.assign_parameters(
                    {self.params_vec: self.obs_params}
                )
                ################# EDITED END

                count = 0
                self.overlap = [0.01, 0]
                g_norm = 1

                if opt == "adam":
                    m = np.zeros(len(self.parameters))
                    v = np.zeros(len(self.parameters))

                if opt == "momentum":
                    old_grad = np.zeros(len(self.parameters))
                    g = np.zeros((len(self.parameters), 2))

                while self.overlap[0] < ths and count < max_iter:
                    # print("Shift optimizing step:", count + 1)
                    count = count + 1

                    if opt == "momentum":
                        old_grad = np.asarray(g[:, 0])
                    ## Measure energy and gradient

                    if grad == "param_shift":
                        E, g = self.compute_overlap_and_gradient(
                            state_wfn, self.parameters, self.shift, expectation, sampler
                        )
                    if grad == "spsa":
                        E, g = self.compute_overlap_and_gradient_spsa(
                            state_wfn,
                            self.parameters,
                            self.shift,
                            expectation,
                            sampler,
                            count,
                        )

                    tot_steps = tot_steps + 1

                    if count == 1:
                        initial_fidelities.append(self.overlap[0])
                        err_init_fid.append(self.overlap[1])

                    # print("Overlap", self.overlap)
                    # print("Gradient", self.gradient[:, 0])

                    if opt == "adam":
                        print("\n Adam \n")
                        meas_grad = np.asarray(g[:, 0])
                        self.shift = np.asarray(
                            self.adam_gradient(count, m, v, meas_grad)
                        )

                    if opt == "momentum":
                        print("Momentum")
                        m_grad = np.asarray(g[:, 0]) + 0.9 * old_grad
                        self.shift = self.shift + m_grad

                    elif opt == "sgd":
                        self.shift = self.shift + g[:, 0]

                    # Norm of the gradient
                    g_vec = np.asarray(g[:, 0])
                    g_norm = np.linalg.norm(g_vec)

                self.parameters = self.parameters + self.shift

                qc_WF = np.asarray(
                    self.instance_statevector.backend.run(
                        self.base_ansatz(
                            np.int(np.log2(log_data["res"])),
                            log_data["depth"],
                            self.parameters,
                        )
                    )
                    .result()
                    .get_statevector()
                )

                H0 = prep_ham(
                    positions[-1],
                    r_f=initial_parameters["r_f"],
                    r_l=initial_parameters["r_l"],
                    r_r=initial_parameters["r_r"],
                    resolution=initial_parameters["res"],
                    L=initial_parameters["L"],
                    padding=initial_parameters["padding"],
                )
                log_data["QC_energy_Vnuc"].append(H0[6][0])
                log_data["fidelity_to_ideal"].append(
                    np.abs(np.inner(np.conjugate(qc_WF), ideal.store_psi[i + 1])) ** 2
                )
                log_data["fidelity_to_exact"].append(
                    np.abs(np.inner(np.conjugate(qc_WF), exact.store_psi[i + 1])) ** 2
                )
                log_data["QC_coefficients"].append((np.abs(np.conj(H0[2]).T @ qc_WF) ** 2).tolist())
                # Measure quantities and save them

                if len(obs_dict) > 0:

                    for (obs_name, obs_pauli) in obs_dict.items():
                        run_measure = self.measure_aux_ops(
                            obs_wfn,
                            obs_pauli(
                                r_nuc=positions[-1],
                                r_f=initial_parameters["r_f"],
                                r_l=initial_parameters["r_l"],
                                r_r=initial_parameters["r_r"],
                                resolution=initial_parameters["res"],
                                L=initial_parameters["L"],
                                padding=initial_parameters["padding"],
                            ),
                            self.parameters,
                            expectation,
                            sampler,
                        )
                        obs_measure[str(obs_name)].append(run_measure[0])
                        obs_error["err_" + str(obs_name)].append(run_measure[1])

                counter.append(count)
                fidelities.append(self.overlap[0])
                err_fin_fid.append(self.overlap[1])

                force_el = -obs_measure["force"][-1] / 1836  # f_iel, 1st f_2el
                force_nuc = (
                    -initial_parameters["nuc_force"](
                        positions[-1],
                        r_f=initial_parameters["r_f"],  # f_inuc, 1st #f2_nuc
                        r_l=initial_parameters["r_l"],
                        r_r=initial_parameters["r_r"],
                        resolution=initial_parameters["res"],
                        L=initial_parameters["L"],
                        padding=initial_parameters["padding"],
                    )
                    / 1836
                )[0]

                if initial_parameters["parameterized"]:
                    force = log_data["ideal_tot_forces"][i+1]
                else:
                    force = force_nuc + force_el  # fi, 1st f2
                forces_el.append(force_el)
                forces_nuc.append(force_nuc)
                forces.append(force)
                velocities.append(velocities[-1] + dt * ((forces[-1] + forces[-2]) / 2))
                log_data["QC_energy_el"].append(is_real(inner(qc_WF, H0[0])))
                log_data["QC_energy_Tnuc"].append(
                    velocities[-1] * velocities[-1] * 0.5 * 1836
                )
                log_data["QC_energy_Vnuc"].append(H0[6][0])
                log_data["QC_energy"].append(log_data["QC_energy_el"][-1]+log_data["QC_energy_Tnuc"][-1]+log_data["QC_energy_Vnuc"][-1])
                if i%50==0:
                    if len(obs_dict) > 0:
                        for (obs_name, obs_pauli) in obs_dict.items():
                            log_data[str(obs_name)] = obs_measure[str(obs_name)]
                            log_data["err_" + str(obs_name)] = obs_error["err_" + str(obs_name)]
                    log_data["init_F"] = initial_fidelities
                    log_data["QC_positions"] = positions
                    log_data["QC_velocities"] = velocities
                    log_data["QC_tot_forces"] = forces
                    log_data["QC_forces_nuc"] = forces_nuc
                    log_data["QC_forces_el"] = forces_el
                    log_data["final_F"] = fidelities
                    log_data["err_init_F"] = err_init_fid
                    log_data["err_fin_F"] = err_fin_fid
                    log_data["iter_number"] = counter
                    log_data["times"] = times
                    log_data["params"] = list(params)
                    log_data["tot_steps"] = [tot_steps]
                    log_data["final_i"] = i
                    json.dump(log_data, open(filename+'chk', "w+"))
        finally:
            print("successfully entered writing routine")

            if len(obs_dict) > 0:
                for (obs_name, obs_pauli) in obs_dict.items():
                    log_data[str(obs_name)] = obs_measure[str(obs_name)]
                    log_data["err_" + str(obs_name)] = obs_error["err_" + str(obs_name)]

            ### QC results
            log_data["init_F"] = initial_fidelities
            log_data["QC_positions"] = positions
            log_data["QC_velocities"] = velocities
            log_data["QC_tot_forces"] = forces
            log_data["QC_forces_nuc"] = forces_nuc
            log_data["QC_forces_el"] = forces_el
            log_data["final_F"] = fidelities
            log_data["err_init_F"] = err_init_fid
            log_data["err_fin_F"] = err_fin_fid
            log_data["iter_number"] = counter
            log_data["times"] = times
            log_data["params"] = list(params)
            log_data["tot_steps"] = [tot_steps]
            log_data["final_i"] = i

            json.dump(log_data, open(filename, "w+"))
            print("Succesfully saved everything")
