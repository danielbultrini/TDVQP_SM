from pVQD_cluster import pVQD
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.states import statevector
from qiskit.opflow.primitive_ops import PauliSumOp
from qiskit.utils import QuantumInstance
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import L_BFGS_B
from ansatze import hweff_ansatz, hweff_param_ansatz
from qiskit import Aer
from scipy.sparse import diags
import shin_metiu_full as sf
import numpy as np
import scipy.linalg as sl
import argparse
import json
from auto_plotter import autoplotter
import warnings

warnings.filterwarnings("ignore")

CLI = argparse.ArgumentParser()
CLI.add_argument("-n", dest="name", type=str, default="test")
CLI.add_argument("-r", dest="resolution", type=int, default=16)
CLI.add_argument("-c", dest="cost", type=str, default="local")
CLI.add_argument("-o", dest="opt", type=str, default="sgd")
CLI.add_argument("-g", dest="grad", type=str, default="param_shift")
CLI.add_argument("-rs", dest="restart", type=str, default="no")
CLI.add_argument("-t", dest="dt", type=float, default=0.05)
CLI.add_argument("-tr", dest="trotter_steps", type=int, default=1)
CLI.add_argument("-s", dest="n_steps", type=int, default=1000)
CLI.add_argument("-d", dest="depth", type=int, default=3)
CLI.add_argument("-rf", dest="r_f", type=float, default=5)
CLI.add_argument("-rl", dest="r_l", type=float, default=4)
CLI.add_argument("-rr", dest="r_r", type=float, default=3.2)
CLI.add_argument("-sh", dest="shots", type=int, default=10000)
CLI.add_argument("-L", dest="L", type=float, default=19)
CLI.add_argument("-ths", dest="ths", type=float, default=0.999999)
CLI.add_argument("-x", dest="x_0", type=float, default=-2)
CLI.add_argument("-v", dest="v_0", type=float, default=0.00114)
CLI.add_argument("-i", dest="initial_state", type=int, nargs="*", default=[0])
CLI.add_argument("-p", dest="parameterized", type=int, default=0)
CLI.add_argument("-pad", dest="padding", type=int, default=20)
CLI.add_argument("-m", dest="md", type=int, default=0)
CLI.add_argument("-b", dest="backend", type=str, default= "statevector_simulator")


args = vars(CLI.parse_args())


def pvqd_experiment(
    shots=10000,
    opt="sgd",
    grad="param_shift",
    cost="local",
    dt=0.05,
    n_steps=1000,
    ths=0.999999,
    depth=3,
    r_f=5,
    r_l=4,
    r_r=3.2,
    x_0=-2,
    v_0=0.00114,
    resolution=16,
    L=19,
    name="test",
    trotter_steps=1,
    initial_state=[0],
    parameterized=0,
    padding=20,
    restart="no",
    md = 0,
    backend = "statevector_simulator"
):
    def prepare_state(r_f, r_l, r_r, resolution=resolution, L=L, padding=padding):
        full = sf.ShinMetiu(
            nstates=resolution,
            L=L,
            Rf=r_f,
            Rl=r_l,
            Rr=r_r,
            mass=1836.0,
            m_el=1.0,
            nel=resolution,
            padding=padding,
        )
        steps = resolution
        dx = L / (steps + 1)
        L2 = L / 2 - dx
        xgrid = np.linspace(-L2, L2, steps)
        return full, xgrid

    def prep_ham(r_nuc, r_f, r_l, r_r, resolution=resolution, L=L, padding=padding):
        full, xgrid = prepare_state(r_f, r_l, r_r, resolution, L=L, padding=padding)
        H_diag, H_off_diag = full.H(np.array([r_nuc]))
        V_nuc = full.V_nuc(np.array([r_nuc]))
        dHdr = full.dV_el(np.array([r_nuc]))
        dHdr_nuc = full.dV_nuc(np.array([r_nuc]))
        En, Q = sl.eigh_tridiagonal(H_diag, H_off_diag, eigvals_only=False)
        H = diags([H_diag, H_off_diag, H_off_diag], [0, 1, -1]).toarray()
        return H, En, Q, dHdr, xgrid, dHdr_nuc, V_nuc

    def hamiltonian(
        r_nuc=x_0,
        r_f=r_f,
        r_l=r_l,
        r_r=r_r,
        resolution=resolution,
        L=L,
        padding=padding,
    ):
        full, xgrid = prepare_state(r_f, r_l, r_r, resolution, L=L, padding=padding)
        H_diag, H_off_diag = full.H_el(np.array([r_nuc]))
        H = diags([H_diag, H_off_diag, H_off_diag], [0, 1, -1]).toarray()
        return PauliSumOp(SparsePauliOp.from_operator(H))

    def el_force(
        r_nuc=x_0,
        r_f=r_f,
        r_l=r_l,
        r_r=r_r,
        resolution=resolution,
        L=L,
        padding=padding,
    ):
        full, xgrid = prepare_state(r_f, r_l, r_r, resolution, L=L, padding=padding)
        dH = full.dV_el(np.array([r_nuc]))
        return PauliSumOp(SparsePauliOp.from_operator(dH))

    def nuc_force(
        r_nuc=x_0,
        r_f=r_f,
        r_l=r_l,
        r_r=r_r,
        resolution=resolution,
        L=L,
        padding=padding,
    ):
        full, xgrid = prepare_state(r_f, r_l, r_r, resolution, L=L, padding=padding)
        dH = full.dV_nuc(np.array([r_nuc]))
        return dH
    def run(    shots,
                opt,
                grad,
                cost,
                dt,
                n_steps,
                ths,
                depth,
                r_f,
                r_l,
                r_r,
                x_0,
                v_0,
                resolution,
                L,
                name,
                trotter_steps,
                initial_state,
                parameterized,
                padding,
                restart,
                backend
):
        backend = Aer.get_backend(backend)
        instance = QuantumInstance(backend=backend, shots=shots, max_parallel_threads=1)
        obs = {"force": el_force, "energy": hamiltonian}
        if parameterized == 0:
            parameterized = False
        elif parameterized == 1:
            parameterized = True


        if restart == "no":
            all = prep_ham(x_0, r_f, r_l, r_r, resolution, L, padding=padding)
            H = all[0]

            dH = all[3]
            dH = PauliSumOp(SparsePauliOp.from_operator(dH))
            H_2 = PauliSumOp(SparsePauliOp.from_operator(H))

            ex_params = np.zeros((depth + 1) * H_2.num_qubits + depth * (H_2.num_qubits - 1))
            shift = np.array(len(ex_params) * [0.01])
            algo = pVQD(
                hamiltonian=hamiltonian,
                ansatz=hweff_ansatz,
                ansatz_reps=depth,
                parameters=ex_params,
                initial_shift=shift,
                instance=instance,
                shots=shots,
            )

            vector = np.zeros(len(all[2][0]))
            for i in initial_state:
                vector += all[2][:, i]  # [::-1]
            vector = vector / np.linalg.norm(vector)
            init_state = statevector.Statevector(vector)

            if initial_state != [0]:
                initial_point = (np.random.random(len(ex_params)) - 0.5) * 4 * np.pi
                print("Random initial parameters:")
                print(initial_point)
                initial_point = algo.prepare_arb_state(
                    init_state, initial_point, max_iter=300, ths=0.99999
                )
            else:
                # initial_point = (np.random.random(len(ex_params))-0.5)*4*np.pi
                print("initial states: ", initial_state)
                print("Finding ground state")
                ansatz = hweff_param_ansatz(H_2.num_qubits, depth)
                vqe = VQE(ansatz, L_BFGS_B(maxiter=500), quantum_instance=instance)
                result = vqe.compute_minimum_eigenvalue(operator=H_2)
                initial_point = np.array(list(result.optimal_parameters.values()))
                print("VQE energy: ", result.eigenvalue)
                print("exa energy: ", all[1][0])
                print("optimizing GS further:")
                print(initial_point)
                initial_point = algo.prepare_arb_state(
                    init_state, initial_point, max_iter=150, ths=0.99999
                )

        else:
            print(f"restarting from {restart}")
            with open(restart) as data:
                d = json.load(data)
                initial_point = np.array(d["params"][0])
                dt = d['dt']
                r_f = d["r_f"]
                r_l = d["r_l"]
                r_r = d["r_r"]
                x_0 = d["ideal_positions"][0]
                v_0 = d["ideal_velocities"][0]
                resolution = d["res"]
                L = d["L"]
                trotter_steps = d['trotter_steps']
                padding = d['padding']

            all = prep_ham(x_0, r_f, r_l, r_r, resolution, L, padding=padding)
            H = all[0]

            dH = all[3]
            dH = PauliSumOp(SparsePauliOp.from_operator(dH))
            H_2 = PauliSumOp(SparsePauliOp.from_operator(H))

            ex_params = np.zeros((depth + 1) * H_2.num_qubits + depth * (H_2.num_qubits - 1))
            shift = np.array(len(ex_params) * [0.01])
            algo = pVQD(
                hamiltonian=hamiltonian,
                ansatz=hweff_ansatz,
                ansatz_reps=depth,
                parameters=ex_params,
                initial_shift=shift,
                instance=instance,
                shots=shots,
            )

            vector = np.zeros(len(all[2][0]))
            for i in initial_state:
                vector += all[2][:, i]  # [::-1]
            vector = vector / np.linalg.norm(vector)
            init_state = statevector.Statevector(vector)


        algo.run(
            ths,
            dt,
            n_steps,
            obs_dict=obs,
            filename=name + ".json",
            max_iter=100,
            opt=opt,
            cost_fun=cost,
            grad=grad,
            initial_point=initial_point,
            initial_parameters={
                "position": x_0,
                "velocity": v_0,
                "nuc_force": nuc_force,
                "r_r": r_r,
                "r_l": r_l,
                "r_f": r_f,
                "L": L,
                "res": resolution,
                "dt": dt,
                "vector": vector,
                "parameterized": parameterized,
                "padding": padding,
            },
            trotter_steps=trotter_steps,
        )
    if md == 0:
        run(    shots,
                opt,
                grad,
                cost,
                dt,
                n_steps,
                ths,
                depth,
                r_f,
                r_l,
                r_r,
                x_0,
                v_0,
                resolution,
                L,
                name,
                trotter_steps,
                initial_state,
                parameterized,
                padding,
                restart,
                backend)
    else: 
        base_name = name
        # generate two lists of random values with normal distributions
        vel_dis = sf.choose_md_velocities()

        # create a list to store the random combinations
        combinations = []

        # generate 5 random combinations of values from the two lists
        for i in range(md):
            # randomly choose one value from each list
            value1 = x_0
            value2 = np.random.choice(vel_dis)
            # add the two values to the combinations list
            combinations.append((value1, value2))
        for ind,comb in enumerate(combinations):
            x_0 = comb[0]
            v_0 = comb[1]
            name = base_name + str(ind)
            run(shots,
                opt,
                grad,
                cost,
                dt,
                n_steps,
                ths,
                depth,
                r_f,
                r_l,
                r_r,
                x_0,
                v_0,
                resolution,
                L,
                name,
                trotter_steps,
                initial_state,
                parameterized,
                padding,
                restart,
                backend,
)

    
        
pvqd_experiment(**args)
# if args['md'] == 0:
#     autoplotter(args["name"] + ".json")
# else:
#     for i in range(args['md']):
#         autoplotter(args["name"]+str(i) + ".json")
 
