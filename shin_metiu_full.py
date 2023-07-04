import copy as cp
import numpy as np
from typing import Tuple, Any, Type, Union
from scipy.special import erf
from scipy.interpolate import interp1d as interp

ArrayLike = Type[np.ndarray]
DtypeLike = Union[np.float64, np.complex128]



def choose_md_velocities(T=300,kB = 3.1668114e-6, mass =1836):
    """Return random choice of velocities from boltzmann distribution, defaults to 300K and atomic units"""
    def MB_CDF(v,m,T):
        """ Cumulative Distribution function of the Maxwell-Boltzmann speed distribution """
        a = np.sqrt(kB*T/m)
        return erf(v/(np.sqrt(2)*a)) - np.sqrt(2/np.pi)* v* np.exp(-v**2/(2*a**2))/a

    # create CDF
    vs = np.arange(0,0.01,0.00001)
    cdf = MB_CDF(vs,mass,T) # essentially y = f(x)

    #create interpolation function to CDF
    inv_cdf = interp(cdf,vs) # essentially what we have done is made x = g(y) from y = f(x)
                            # this can now be used as a function which is 
                            # called in the same way as normal routines

    # create CDF
    vs = np.arange(0,0.01,0.00001)
    cdf = MB_CDF(vs,mass,T) # essentially y = f(x)

    #create interpolation function to CDF
    inv_cdf = interp(cdf,vs) # essentially what we have done is made x = g(y) from y = f(x)

    def generate_velocities(n):
        """ generate a set of velocity vectors from the MB inverse CDF function """
        rand_nums = np.random.random(n)
        return inv_cdf(rand_nums)
    
    return generate_velocities(100000)


class ElectronicModel_(object):
    """
    Base class for handling electronic structure part of dynamics
    """

    def __init__(self, representation: str = "adiabatic", reference: Any = None):
        self.representation = representation
        self.position: ArrayLike
        self.reference = reference

        self.hamiltonian: ArrayLike
        self.force: ArrayLike
        self.derivative_coupling: ArrayLike

    def compute(
        self,
        X: ArrayLike,
        couplings: Any = None,
        gradients: Any = None,
        reference: Any = None,
    ) -> None:
        """
        Central function for model objects. After the compute function exists, the following
        data must be provided:
          - self.hamiltonian -> n x n array containing electronic hamiltonian
          - self.force -> n x ndim array containing the force on each diagonal
          - self.derivative_coupling -> n x n x ndim array containing derivative couplings

        The couplings and gradients options are currently unimplemented, but are
        intended to allow specification of which couplings and gradients are needed
        so that computational cost can be reduced.

        Nothing is returned, but the model object should contain all the
        necessary date.
        """
        raise NotImplementedError("ElectronicModels need a compute function")

    def update(
        self, X: ArrayLike, couplings: Any = None, gradients: Any = None
    ) -> "ElectronicModel_":
        """
        Convenience function that copies the present object, updates the position,
        calls compute, and then returns the new object
        """
        out = cp.copy(self)
        out.position = X
        out.compute(
            X, couplings=couplings, gradients=gradients, reference=self.reference
        )
        return out

    def as_dict(self):
        out = {
            "nstates": self.nstates(),
            "ndim": self.ndim(),
            "position": self.position.tolist(),
            "hamiltonian": self.hamiltonian.tolist(),
            "derivative_coupling": self.derivative_coupling.tolist(),
            "force": self.force.tolist(),
        }
        return out


class AdiabaticModel_(ElectronicModel_):
    """
    Base class to handle model problems that have an auxiliary electronic problem admitting
    many electronic states that are truncated to just a few. Sort of a truncated DiabaticModel_.
    """

    nstates_: int
    ndim_: int

    def __init__(self, representation: str = "adiabatic", reference: Any = None):
        if representation == "diabatic":
            raise Exception("Adiabatic models can only be run in adiabatic mode")
        ElectronicModel_.__init__(
            self, representation=representation, reference=reference
        )

    def nstates(self) -> int:
        return self.nstates_

    def ndim(self) -> int:
        return self.ndim_

    def compute(
        self,
        X: ArrayLike,
        couplings: Any = None,
        gradients: Any = None,
        reference: Any = None,
    ) -> None:
        self.position = X

        self.reference, self.hamiltonian = self._compute_basis_states(
            self.H(X), reference=reference
        )
        dV = self.dV(X)

        self.derivative_coupling = self._compute_derivative_coupling(
            self.reference, dV, np.diag(self.hamiltonian)
        )

        self.force = self._compute_force(dV, self.reference)

        self.force_matrix = self._compute_force_matrix(dV, self.reference)

    def update(
        self, X: ArrayLike, couplings: Any = None, gradients: Any = None
    ) -> "AdiabaticModel_":
        out = cp.copy(self)
        out.position = X
        out.compute(
            X, couplings=couplings, gradients=gradients, reference=self.reference
        )
        return out

    def _compute_basis_states(
        self, V: ArrayLike, reference: Any = None
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Computes coefficient matrix for basis states
        if a diabatic representation is chosen, no transformation takes place
        :param V: potential matrix
        :param reference: ElectronicStates from previous step used only to fix phase
        """
        if self.representation == "adiabatic":
            en, co = np.linalg.eigh(V)
            nst = self.nstates()
            coeff = co[:, :nst]
            energies = en[:nst]

            if reference is not None:
                try:
                    for mo in range(self.nstates()):
                        if np.dot(coeff[:, mo], reference[:, mo]) < 0.0:
                            coeff[:, mo] *= -1.0
                except:
                    raise Exception(
                        "Failed to regularize new ElectronicStates from a reference object %s"
                        % (reference)
                    )
            return coeff, np.diag(energies)
        elif self.representation == "diabatic":
            raise Exception("Adiabatic models can only be run in adiabatic mode")
            return None
        else:
            raise Exception("Unrecognized representation")

    def _compute_force(self, dV: ArrayLike, coeff: ArrayLike) -> ArrayLike:
        r""":math:`-\langle \phi_{\mbox{state}} | \nabla H | \phi_{\mbox{state}} \rangle`"""
        nst = self.nstates()
        ndim = self.ndim()

        half = np.einsum("xij,jp->ipx", dV, coeff)

        out = np.zeros([nst, ndim], dtype=np.float64)
        for ist in range(self.nstates()):
            out[ist, :] += -np.einsum("i,ix->x", coeff[:, ist], half[:, ist, :])
        return out

    def _compute_force_matrix(self, dV: ArrayLike, coeff: ArrayLike) -> ArrayLike:
        r"""returns :math:`F^\xi{ij} = \langle \phi_i | -\nabla_\xi H | \phi_j\rangle`"""
        out = -np.einsum("ip,xij,jq->pqx", coeff, dV, coeff)
        return out

    def _compute_derivative_coupling(
        self, coeff: ArrayLike, dV: ArrayLike, energies: ArrayLike
    ) -> ArrayLike:
        r"""returns :math:`\phi_{i} | \nabla_\alpha \phi_{j} = d^\alpha_{ij}`"""
        if self.representation == "diabatic":
            return np.zeros(
                [self.nstates(), self.nstates(), self.ndim()], dtype=np.float64
            )

        out = np.einsum("ip,xij,jq->pqx", coeff, dV, coeff)

        for j in range(self.nstates()):
            for i in range(j):
                dE = energies[j] - energies[i]
                if abs(dE) < 1.0e-14:
                    dE = np.copysign(1.0e-14, dE)

                out[i, j, :] /= dE
                out[j, i, :] /= -dE

        return out

    def H(self, X: ArrayLike) -> ArrayLike:
        raise NotImplementedError("Diabatic models must implement the function V")

    def dV(self, X: ArrayLike) -> ArrayLike:
        raise NotImplementedError("Diabatic models must implement the function dV")


class ShinMetiu(AdiabaticModel_):
    ndim_: int = 1

    def __init__(
        self,
        representation: str = "adiabatic",
        reference: Any = None,
        nstates: int = 3,
        L: float = 19.0,
        Rf: float = 5.0,
        Rl: float = 3.1,
        Rr: float = 4.0,
        mass: float = 1836.0,
        m_el: float = 1,
        nel: int = 128,
        box: Any = None,
        padding=10,
    ):
        """Constructor defaults to classic Shin-Metiu as described in
        Gossel, Liacombe, Maitra JCP 2019"""
        AdiabaticModel_.__init__(
            self, representation=representation, reference=reference
        )

        self.L = L
        self.ion_left = self.L * 0.5
        self.ion_right = self.L * 0.5
        self.Rf = Rf
        self.Rl = Rl
        self.Rr = Rr
        self.mass = np.array(mass, dtype=np.float64).reshape(self.ndim())
        self.m_el = m_el

        if box is None:
            box = L + padding

        box_left, box_right = -0.5 * box, 0.5 * box
        self.rr = np.linspace(
            box_left + 1e-8, box_right - 1e-8, nel, endpoint=True, dtype=np.float64
        )

        self.nstates_ = nstates

    def soft_coulomb(self, r12: ArrayLike, gamma: DtypeLike) -> ArrayLike:
        abs_r12 = np.abs(r12)
        sc = erf(abs_r12 / gamma) / abs_r12
        if np.any(np.isnan(sc)):
            sc = np.nan_to_num(sc, nan=2 / np.sqrt((gamma**2) * np.pi))
        return sc

    def d_soft_coulomb(self, r12: ArrayLike, gamma: DtypeLike) -> ArrayLike:
        abs_r12 = np.abs(r12)
        two_over_root_pi = 2.0 / np.sqrt(np.pi)
        out = r12 * erf(abs_r12 / gamma) / (
            abs_r12**3
        ) - two_over_root_pi * r12 * np.exp(-(abs_r12**2) / (gamma**2)) / (
            gamma * abs_r12 * abs_r12
        )
        if np.any(np.isnan(out)):
            out = np.nan_to_num(out, 0)

        return out

    def V_nuc(self, R: ArrayLike) -> ArrayLike:
        v0 = 1.0 / np.abs(self.L / 2 - R) + 1.0 / np.abs(self.L / 2 + R)
        if np.isinf(v0) or v0 > 10000:
            return 10000
        return v0

    def vv_el(self, R: ArrayLike) -> ArrayLike:
        rr = self.rr
        v_en = self.soft_coulomb(R - rr, self.Rf)
        v_le = self.soft_coulomb(rr + self.L / 2, self.Rl)
        v_re = self.soft_coulomb(rr - self.L / 2, self.Rr)
        vv = -v_en - v_le - v_re
        return vv

    def vv(self, R: ArrayLike) -> ArrayLike:
        contour = np.zeros((len(R), len(self.rr)))
        for i, pos in enumerate(R):
            contour[i] = self.vv_el(pos) + self.V_nuc(pos)
        return contour

    def vvR(self, R: ArrayLike) -> ArrayLike:
        contour = self.vv_el(R) + self.V_nuc(R)
        return contour

    def H_el(self, R: ArrayLike) -> ArrayLike:
        rr = self.rr
        nr = len(self.rr)
        dr = rr[1] - rr[0]

        vv = self.vv_el(R)  # + self.V_nuc(R)

        kinetic_factor = -(0.5 / (self.m_el * dr**2))
        diag = (kinetic_factor * (-2)) + vv
        off_diag = kinetic_factor * np.ones(nr - 1, dtype=np.float64)

        return diag, off_diag

    def dV_nuc(self, R: ArrayLike) -> ArrayLike:
        # LmR = np.abs(0.5 * self.L - R)
        # LpR = np.abs(0.5 * self.L + R)
        # dv0 = LmR / np.abs(LmR**3) - LpR / np.abs(LpR**3)

        LmR = 0.5 * self.L - R
        LpR = 0.5 * self.L + R
        dv0 = LmR / np.abs(LmR) ** 3 - LpR / np.abs(LpR) ** 3

        return dv0

    def dV_el(self, R: ArrayLike) -> ArrayLike:
        rr = self.rr
        rR = R - rr
        dvv = self.d_soft_coulomb(rR, self.Rf)  # + self.dV_nuc(R)
        return np.diag(dvv)

    def H(self, R: ArrayLike) -> ArrayLike:
        """:math:`V(x)`"""
        return self.H_el(R)

    def dV(self, R: ArrayLike) -> ArrayLike:
        """:math:`\\nabla V(x)`"""
        return (self.dV_el(R) + self.dV_nuc(R)).reshape([1, len(self.rr), len(self.rr)])
