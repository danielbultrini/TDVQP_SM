import numpy as np
import scipy.linalg as sl
from scipy.sparse import csr_matrix, diags
import shin_metiu_full as sf
import pandas as pd
from scipy import sparse
import warnings
import traceback


def inner(state, op):
    if len(state.shape) == 1:
        return np.dot(np.conjugate(state), sparse.csr_matrix(op) * state)
    else:
        return np.dot(np.conjugate(state.T), sparse.csr_matrix(op) * state)


def is_real(num):
    if np.imag(num) < 1e-12:
        return np.real(num)
    else:
        warnings.warn(f"Non negligible imaginary number {num}", stacklevel=2)
        return np.real(num)


def prepare_state(r_f, r_l, r_r, resolution=500, L=19, padding=10):
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
    xgrid = full.rr
    return full, xgrid


def prep_ham(r_nuc, r_f, r_l, r_r, resolution=500, L=19, padding=10):
    full, xgrid = prepare_state(r_f, r_l, r_r, resolution, L=L, padding=padding)
    H_diag, H_off_diag = full.H(np.array([r_nuc]))
    V_nuc = full.V_nuc(np.array([r_nuc]))
    dHdr = full.dV_el(np.array([r_nuc]))
    dHdr_nuc = full.dV_nuc(np.array([r_nuc]))
    En, Q = sl.eigh_tridiagonal(H_diag, H_off_diag, eigvals_only=False)
    H = diags([H_diag, H_off_diag, H_off_diag], [0, 1, -1])
    return H, En, Q, dHdr, xgrid, dHdr_nuc, V_nuc


class simulation:
    def __init__(
        self,
        dt=0.05,
        steps=1000,
        r_f=1.0,
        r_l=1.0,
        r_r=1.0,
        resolution=100,
        L=19,
        x_0=-9.0,
        v_0=0.0,
        vector=None,
        parameterized=False,
        padding=10,
        obs_error = 0 # multiplicative factor - 0 for no error
    ) -> None:
        self.r_f, self.r_l, self.r_r = r_f, r_l, r_r
        self.L = L
        self.resolution = resolution
        dt = dt
        m = 1836
        self.steps = steps
        self.store_xs = np.zeros(self.steps + 2)
        self.store_t = np.zeros(self.steps + 2)
        self.store_fnuc = np.zeros(self.steps + 2)
        self.store_fel = np.zeros(self.steps + 2)
        self.store_f = np.zeros(self.steps + 2)
        self.store_v = np.zeros(self.steps + 2)
        self.store_c = np.zeros((self.steps + 2, self.resolution))
        self.store_eigen = np.zeros((self.steps + 2, self.resolution), dtype=complex)
        self.store_psi = np.zeros((self.steps + 2, self.resolution), dtype=complex)
        self.store_energy_el = np.zeros(self.steps + 2)
        self.store_energy_Tnuc = np.zeros(self.steps + 2)
        self.store_energy_Vnuc = np.zeros(self.steps + 2)

        # Compute initial conditions
        H_0, En_0, Q_0, dHdr_0, self.xgrid, dHdr_nuc0, V_nuc_0 = prep_ham(
            x_0,
            self.r_f,
            self.r_l,
            self.r_r,
            self.resolution,
            L=self.L,
            padding=padding,
        )  # t0
        if vector is not None:
            if type(vector) == list:
                psi = np.zeros(Q_0[:, 0].shape)
                for i in vector:
                    psi += Q_0[:, i]  # [::-1]
                psi_init = psi / np.linalg.norm(psi)

            else:
                psi_init = vector
        else:
            psi_init = Q_0[:, 0]

        psi_init = psi_init / np.linalg.norm(psi_init)

        propagator_0 = Q_0 @ np.diag(np.exp(-1j * En_0 * dt)) @ Q_0.T
        f_0el = -1 * inner(psi_init, dHdr_0) / m
        f_0nuc = -is_real(dHdr_nuc0[0]) / m

        if parameterized:
            f_0 = f_0nuc
        else:
            f_0 = f_0el + f_0nuc
        f_0 += f_0*obs_error
        self.store_energy_el[0] = is_real(inner(psi_init, H_0))
        self.store_energy_Vnuc[0] = V_nuc_0
        self.store_energy_Tnuc[0] = 0.5 * m * (v_0**2)

        self.store_c[0] = np.abs(np.conj(Q_0).T @ psi_init) ** 2

        # first timestep approximation
        x_1 = x_0 + v_0 * dt + 0.5 * (f_0) * dt * dt  # t1
        psi_1 = propagator_0 @ psi_init
        psi_1 = psi_1 / np.linalg.norm(psi_1)
        H_1, En_1, Q_1, dHdr_1, xgrid, dHdr_nuc1, V_nuc_1 = prep_ham(
            x_1,
            self.r_f,
            self.r_l,
            self.r_r,
            self.resolution,
            L=self.L,
            padding=padding,
        )  # t1
        f_1el = is_real(-inner(psi_1, dHdr_1) / m)
        f_1nuc = is_real(-dHdr_nuc1[0] / m)
        if parameterized:
            f_1 = f_1nuc  # t1
        else:
            f_1 = f_1el + f_1nuc
        f_1 += f_1*obs_error
        propagator = Q_1 @ np.diag(np.exp(-1j * En_1 * dt)) @ Q_1.T  # t1
        v_1 = v_0 + dt * (f_0 + f_1) / 2
        self.store_energy_el[1] = is_real(inner(psi_1, H_1))
        self.store_energy_Vnuc[1] = V_nuc_1
        self.store_energy_Tnuc[1] = 0.5 * m * (v_1**2)
        self.store_c[1] = np.abs(np.conj(Q_1).T @ psi_1) ** 2

        # Setup initial conditions
        self.store_fel[0], self.store_fel[1] = is_real(f_0el), is_real(f_1el)
        self.store_fnuc[0], self.store_fnuc[1] = is_real(f_0nuc), is_real(f_1nuc)
        self.store_xs[0], self.store_xs[1] = is_real(x_0), is_real(x_1)
        self.store_f[0], self.store_f[1] = is_real(f_0), is_real(f_1)
        self.store_psi[0], self.store_psi[1] = psi_init, psi_1
        self.store_t[0], self.store_t[1] = 0, dt
        self.store_v[0], self.store_v[1] = is_real(v_0), is_real(v_1)
        self.store_eigen[0], self.store_eigen[1] = En_0, En_1

        for i in range(1, self.steps + 1):
            self.store_xs[i + 1] = is_real(
                self.store_xs[i]
                + self.store_v[i] * dt
                + 0.5 * (self.store_f[i]) * dt * dt
            )
            H, En, Q, dHdr, xgrid, dHdr_nuc, V_nuc = prep_ham(
                self.store_xs[i + 1],
                self.r_f,
                self.r_l,
                self.r_r,
                self.resolution,
                L=self.L,
                padding=padding,
            )
            propagator = Q @ np.diag(np.exp(-1j * En * dt)) @ Q.T
            self.store_psi[i + 1] = propagator @ self.store_psi[i]
            self.store_psi[i + 1] = self.store_psi[i + 1] / np.linalg.norm(
                self.store_psi[i + 1]
            )
            self.store_fel[i + 1] = is_real(-inner(self.store_psi[i + 1], dHdr) / m)
            self.store_fnuc[i + 1] = is_real(-dHdr_nuc[0] / m)
            if parameterized:
                self.store_f[i + 1] = is_real(
                    self.store_fnuc[i + 1]
                )  # +self.store_fel[i+1]
            else:
                self.store_f[i + 1] = is_real(
                    self.store_fel[i + 1] + self.store_fnuc[i + 1]
                )
                self.store_f[i + 1] += self.store_f[i + 1]*obs_error
            self.store_v[i + 1] = is_real(
                self.store_v[i] + dt * (self.store_f[i] + self.store_f[i + 1]) / 2
            )
            self.store_t[i + 1] = self.store_t[i] + dt
            self.store_c[i + 1] = np.abs(np.conj(Q).T @ self.store_psi[i + 1]) ** 2
            self.store_energy_el[i + 1] = is_real(inner(self.store_psi[i + 1], H))
            self.store_energy_Vnuc[i + 1] = V_nuc
            self.store_energy_Tnuc[i + 1] = 0.5 * m * (self.store_v[i + 1] ** 2)

            self.store_eigen[i + 1] = En

        self.coef_df = pd.DataFrame(
            np.abs(self.store_c),  # ** 2,
            columns=range(self.resolution),
            index=range(self.steps + 2),
        )
        self.coef_df = (
            self.coef_df.rename_axis("time")
            .reset_index()
            .melt("time", var_name="state")
        )
        self.map_xs = {
            list(range(self.steps + 2))[i]: self.store_xs[i]
            for i in range(self.steps + 2)
        }
        self.map_ts = {
            list(range(self.steps + 2))[i]: self.store_t[i]
            for i in range(self.steps + 2)
        }
        self.coef_df["position"] = self.coef_df["time"].map(self.map_xs)
        self.coef_df["time"] = self.coef_df["time"].map(self.map_ts)
        self.store_eigen_pos = np.zeros((len(self.xgrid), self.resolution))
        for i in range(len(self.xgrid)):
            H, En, Q, dHdr, xgrid, dHdr_nuc, V_nuc = prep_ham(
                self.xgrid[i],
                self.r_f,
                self.r_l,
                self.r_r,
                self.resolution,
                L=self.L,
                padding=padding,
            )
            self.store_eigen_pos[i] = En
        
