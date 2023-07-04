import json
import os
from qiskit import Aer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ansatze import hweff_ansatz
from exact_simulator import prep_ham, simulation, inner, is_real


def autoplotter(file):
    """A plotting helper function that generates a variety of plots in a folder
    with the filename of the data file which will be filled with all plots.

    Args:
        file (json file location): json file with simulation results
    """
    save_loc = os.path.dirname(file) + "/plots/"
    filename = os.path.splitext(os.path.basename(file))[0]

    try:
        os.mkdir(save_loc)
    except:
        pass

    save_loc += filename + "/"
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    save_loc += filename
    data = json.load(open(file))
    backend = Aer.get_backend("statevector_simulator")

    def plot(p):
        fig = plt.figure()
        l = len(data["times"])
        gs = fig.add_gridspec(2, hspace=0.1, height_ratios=[2, 1])
        axs = gs.subplots(sharex=True)
        fig.suptitle(p)
        axs[1].plot(
            np.abs(np.array(data[f"QC_{p}"])[0:l] - np.array(data[f"exact_{p}"])[0:l]),
            color="red",
            label="difference",
        )
        axs[0].plot(
            (np.array(data[f"QC_{p}"]))[0:l],
            color="pink",
            label="Exact from ideal state",
        )
        axs[0].plot(
            (np.array(data[f"QC_{p}"]))[0:l], color="green", label="QC", dashes=[3, 1]
        )

        if p == "QC_forces_el":
            axs[0].plot(
                t.store_fel, color="black", label="Exact from QC state", dashes=[4, 3]
            )[0:l]
        # axs[0].set_yscale('log')
        axs[0].legend()
        axs[1].legend(loc=4)
        # axs[0].set_yscale('log')
        axs[1].set_yscale("log")
        plt.tight_layout()
        plt.savefig(save_loc + f"_plot_{p}.svg")
        plt.clf()

    starting_state = np.asarray(
        backend.run(
            hweff_ansatz(np.int(np.log2(data["res"])), data["depth"], data["params"][0])
        )
        .result()
        .get_statevector()
    )
    t = simulation(
        dt=data["dt"],
        steps=len(data["QC_positions"]) - 3,
        r_f=data["r_f"],
        r_l=data["r_l"],
        r_r=data["r_r"],
        resolution=data["res"],
        L=data["L"],
        x_0=data["QC_positions"][0],
        v_0=data["QC_velocities"][0],
        vector=starting_state,
        parameterized=data["parameterized"],
        padding=data["padding"],
    )
    store_c = np.zeros((len(data["params"]), data["res"]))
    store_fel = np.zeros(len(data["params"]), dtype=np.complex128)
    QC_fel = np.zeros(len(data["params"]), dtype=np.complex128)
    fidelity = []

    for i, val in enumerate(data["params"]):
        starting_state = np.asarray(
            backend.run(hweff_ansatz(np.int(np.log2(data["res"])), data["depth"], val))
            .result()
            .get_statevector()
        )
        H, En, Q, dHdr, xgrid, dHdr_nuc, V_nuc = prep_ham(
            data["QC_positions"][i],
            data["r_f"],
            data["r_l"],
            data["r_r"],
            data["res"],
            data["L"],
            padding=data["padding"],
        )
        store_c[i] = np.abs(np.conj(Q).T @ starting_state) ** 2
        store_fel[i] = -1 * inner(t.store_psi[i], dHdr) / 1836
        QC_fel[i] = -1 * inner(starting_state, dHdr) / 1836
        fidelity.append(
            np.abs(np.inner(np.conjugate(starting_state), t.store_psi[i])) ** 2
        )

    df = pd.DataFrame(
        store_c, columns=range(data["res"]), index=range(len(data["params"]))
    )
    df = df.rename_axis("time").reset_index().melt("time", var_name="state")
    df["time"] = df["time"] * data["dt"]
    df["position"] = data["QC_positions"][0:-1] * data["res"]
    df["velocity"] = data["QC_velocities"][0:-1] * data["res"]

    for value in [
        "positions",
        "tot_forces",
        "forces_el",
        "forces_nuc",
        "velocities",
        "energy",
    ]:
        plot(value)

    sns.lineplot(
        data=t.coef_df.query("state<4 & value>1e-5").abs(),
        x="time",
        y="value",
        hue="state",
        style="state",
    ).set(
        yscale="log",
        ylabel="Population",
        xlabel="Time [Arb]",
        title="Exact evolution",
        ylim=[1e-5, 5],
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig(save_loc + "_state_plot_sim.svg")
    plt.clf()

    sns.lineplot(
        data=df.query("state<4 & value>1e-5"),
        x="time",
        y="value",
        hue="state",
        style="state",
    ).set(
        yscale="log",
        ylabel="Population",
        xlabel="Time [Arb]",
        title="QC evolution",
        ylim=[1e-5, 5],
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig(save_loc + "_state_plot_QC.svg")
    plt.clf()

    plt.plot(data["times"], fidelity[0 : len(data["times"])])
    plt.ylabel("Fidelity")
    plt.xlabel("Time [arb.]")
    plt.tight_layout()
    plt.savefig(save_loc + "_fidelity.svg")
    plt.clf()

    sim = simulation(
        1, 1, data["r_f"], data["r_l"], data["r_r"], data["res"], data["L"]
    )
    En_df = pd.DataFrame(
        sim.store_eigen_pos, columns=range(sim.resolution), index=range(len(sim.xgrid))
    )
    map_xs = {
        list(range(sim.resolution))[i]: sim.xgrid[i] for i in range(sim.resolution)
    }
    En_df = (
        En_df.rename_axis("position").reset_index().melt("position", var_name="Energy")
    )
    En_df["position"] = En_df["position"].map(map_xs)
    sns.lineplot(
        data=En_df.query("Energy<4"), x="position", y="value", hue="Energy"
    ).set(ylabel="Energy [Arb]", xlabel="Position [Arb]")
    plt.xlim()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.tight_layout()
    plt.savefig(save_loc + "_PES.svg")
    plt.clf()


def prep_dfs(file):
    data = json.load(open(file))
    backend = Aer.get_backend("statevector_simulator")

    res = np.asarray(
        backend.run(
            hweff_ansatz(np.int(np.log2(data["res"])), data["depth"], data["params"][0])
        )
        .result()
        .get_statevector()
    )
    t = simulation(
        data["dt"],
        len(data["positions"]) - 3,
        data["r_f"],
        data["r_l"],
        data["r_r"],
        data["res"],
        data["L"],
        x_0=data["positions"][0],
        v_0=data["velocities"][0],
        padding=data['padding'],
        vector=res,
        parameterized=data["parameterized"],
    )
    store_c = np.zeros((len(data["params"]), data["res"]))
    store_fel = np.zeros(len(data["params"]))
    QC_fel = np.zeros(len(data["params"]))
    fidelity = np.zeros(len(data["params"]))

    for i, val in enumerate(data["params"]):
        res = np.asarray(
            backend.run(hweff_ansatz(np.int(np.log2(data["res"])), data["depth"], val))
            .result()
            .get_statevector()
        )
        H, En, Q, dHdr, xgrid, dHdr_nuc, V_nuc = prep_ham(
            data["QC_positions"][i],
            data["r_f"],
            data["r_l"],
            data["r_r"],
            data["res"],
            data["L"],
            padding=data['padding']
        )
        store_c[i] = np.abs(np.conj(Q).T @ res) ** 2
        store_fel[i] = -1 * is_real(inner(t.store_psi[i], dHdr)) / 1836
        QC_fel[i] = -1 * is_real(inner(res, dHdr)) / 1836
        fidelity[i] = np.abs(np.inner(np.conjugate(res), t.store_psi[i])) ** 2

    df = pd.DataFrame(
        store_c, columns=range(data["res"]), index=range(len(data["params"]))
    )
    df = df.rename_axis("time").reset_index().melt("time", var_name="state")
    df["time"] = df["time"] * data["dt"]
    df["position"] = data["QC_positions"][0:-1] * data["res"]
    df["velocity"] = data["QC_velocities"][0:-1] * data["res"]

    df2 = pd.DataFrame(
        {
            "Exact Electron Force": store_fel,
            "QC Electron Force": QC_fel,
            "Fidelity": fidelity,
            "time": np.array(list(range(len(fidelity)))) * data["dt"],
        }
    )
    df2["Trotter steps"] = data["trotter_steps"]
    df2["dt"] = data["dt"]
    df2[r"$R_r$"] = data["r_r"]
    df2[r"$R_l$"] = data["r_l"]
    df2[r"$R_f$"] = data["r_f"]
    df2[r"$L$"] = data["r_f"]
    df2["Parameterized"] = data["parameterized"]
    df2["Padding"] = data["padding"]
    df2["Depth"] = data["depth"]
    df2["Resolution"] = data["res"]
    df2["file"] = file

    df["Trotter steps"] = data["trotter_steps"]
    df["dt"] = data["dt"]
    df[r"$R_r$"] = data["r_r"]
    df[r"$R_l$"] = data["r_l"]
    df[r"$R_f$"] = data["r_f"]
    df[r"$L$"] = data["r_f"]
    df["Parameterized"] = data["parameterized"]
    df["Padding"] = data["padding"]
    df["Depth"] = data["depth"]
    df["Resolution"] = data["res"]
    df["file"] = file

    return t, df, df2
