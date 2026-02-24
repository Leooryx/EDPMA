"""
Solver for the equation:
h * sum_{j>=1} psi(Z^n - Z^{n-j}) * R_j = f^n

where:
- Z^n = z_p(n*h) for -N <= n < 0 (initial conditions)
- psi(u) = u^alpha (component-wise for vectors)
- f^n = f(n*h)
- Z^n is a vector of dimension d in {1, 2}
"""

# this code represents the dynamic according to the density of the links. 

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from typing import Callable, Tuple, Optional
from tqdm import tqdm


class VolterraSolver:
    """
    Solves the discrete Volterra-type equation with power nonlinearity.
    """
    
    def __init__(
        self,
        h: float,
        delta_x: float,
        N_x: int,
        N: int,
        M: int,
        d: int,
        alpha: float,
        R: np.ndarray,
        f_func: Callable[[float], np.ndarray],
        z_initial: Callable[[float], np.ndarray]
    ):
        """
        Parameters:
        -----------
        h : float
            Time step # delta_t
        delta_x : float
            Space step
        N_x : int
            Number of space points
        N : int
            Number of initial points (for n in [-N, 0)) # points for past history
        M : int
            Number of points to compute (for n in [0, M])
        d : int
            Dimension of Z (1 or 2)
        alpha : float
            Power in psi(u) = u^alpha
        R : np.ndarray
            Coefficients R_j (length should be at least M+N)
        f_func : callable
            Function f(t) returning d-dimensional vector
        z_initial : callable
            Initial condition function z_p(t) for t < 0
        """
        self.h = h
        self.delta_x = delta_x
        self.N_x = N_x
        self.N = N
        self.M = M
        self.d = d
        self.alpha = alpha
        self.R = R
        self.f_func = f_func
        self.z_initial = z_initial
        
        # Storage for solution
        self.Z = np.zeros((N + M + 1, d)) #matrix 
        
        # Initialize with initial conditions
        for n in range(-N, 0): # include past history
            self.Z[n + N] = self.z_initial(n * h) 
    
    def psi(self, u: np.ndarray) -> np.ndarray:
        """
        Nonlinearity psi(u) = u^alpha (component-wise).
        Handles sign for non-integer alpha.
        """
        if self.alpha == int(self.alpha) and int(self.alpha) % 2 == 1:
            # Odd integer power: preserve sign
            return np.sign(u) * np.abs(u) ** self.alpha
        else:
            # Even power or non-integer: use absolute value
            return np.abs(u) ** self.alpha

    # local time ###################################################################
    def local_time(self, n, z):
        local_time = 0
        for i in range(0,n+1): #TODO: pb index, commencer par 1 plutôt ?
            local_time = local_time + self.h * (i*self.delta_x <= z <= (i+1)*self.delta_x).astype(int)
        return local_time

    def residual(self, Z_n: np.ndarray, n: int) -> np.ndarray:
        """
        Compute residual: [Previous: h * sum_{j>=1} psi(Z^n - Z^{n-j}) * R_j - f^n]
        """
        res = np.zeros(self.d)
        
        for k in range(0, self.N_x):
            res = res + self.psi(Z_n - k*self.delta_x) * self.local_time(n, Z_n) * self.delta_x
            # comment intégrer les poids exponentiels ?    #* self.R[j]
        
        # Subtract f^n
        res -= self.f_func(n * self.h)
        
        return res
    
    def solve_step(self, n: int) -> np.ndarray:
        """
        Solve for Z^n using nonlinear solver.
        """
        # Initial guess: use previous value
        Z_guess = self.Z[n + self.N - 1] if n > 0 else self.Z[self.N - 1]
        
        # Solve nonlinear system
        solution = fsolve(
            lambda z: self.residual(z, n),
            Z_guess,
            full_output=False
        )
        
        return solution
    
    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the equation for n = 0, 1, ..., M.
        
        Returns:
        --------
        t : np.ndarray
            Time points
        Z : np.ndarray
            Solution array (shape: (N+M+1, d))
        """
        # Solve for each time step
        for n in tqdm(range(self.M + 1), desc="Solving steps", unit="step"):
            self.Z[n + self.N] = self.solve_step(n)
            
            if (n + 1) % 100 == 0:
                print(f"Solved up to n = {n}")
        
        # Time array
        t = np.array([n * self.h for n in range(-self.N, self.M + 1)])
        
        return t, self.Z
    
    def plot_solution(self, file_name, t: np.ndarray, Z: np.ndarray):
        """
        Plot the solution.
        """
        fig, axes = plt.subplots(self.d, 1, figsize=(10, 4*self.d))
        
        if self.d == 1:
            axes = [axes]
        
        for i in range(self.d):
            axes[i].plot(t, Z[:, i], 'b-', linewidth=1.5, label=f'$Z_{i+1}(t)$')
            axes[i].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='$t=0$')
            axes[i].set_xlabel('$t$')
            axes[i].set_ylabel(f'$Z_{i+1}$')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(f'Solution for $\\psi(u) = u^{{{self.alpha}}}$, $d={self.d}$')
        plt.tight_layout()
        plt.savefig(f'/home/onyxia/work/EDPMA/{file_name}.png')
        plt.show()

# quasi ok niveau compréhension

# ============================================================================
# Functions for f(t)
# ============================================================================

def f_constant(d: int, value: float = 1.0) -> Callable[[float], np.ndarray]:
    """Constant forcing function."""
    return lambda t: value * np.ones(d)


def f_oscillating(d: int, amplitude: float = 1.0, frequency: float = 1.0) -> Callable[[float], np.ndarray]:
    """Oscillating forcing function."""
    if d == 1:
        return lambda t: amplitude * np.array([np.sin(2 * np.pi * frequency * t)])
    else:  # d == 2
        return lambda t: amplitude * np.array([
            np.sin(2 * np.pi * frequency * t),
            np.cos(2 * np.pi * frequency * t)
        ])


def f_random(d: int, amplitude: float = 1.0, seed: Optional[int] = None) -> Callable[[float], np.ndarray]:
    """Random forcing function (uses noise with spatial correlation).""" # how is it spatial??
    rng = np.random.RandomState(seed)
    
    def random_func(t):
        # Use time as seed for reproducibility
        local_rng = np.random.RandomState(int(abs(t * 1000)) % (2**31))
        return amplitude * (2 * local_rng.rand(d) - 1) # only random numbers with no fixed distribution
    
    return random_func


def z_initial_default(d: int, amplitude: float = 0.1) -> Callable[[float], np.ndarray]:
    """Default initial condition: exponential decay as t -> -infinity."""
    return lambda t: amplitude * np.exp(t / 10) * np.ones(d) # the past history, as a function of the time


def compute_R_exponential(length: int, decay_rate: float = 0.5) -> np.ndarray:
    """
    Compute R_j coefficients with exponential decay.
    R_j = exp(-decay_rate * j)
    """
    R = np.zeros(length)
    R[0] = 0  # R_0 is not used
    for j in range(1, length):
        R[j] = np.exp(-decay_rate * j)
    return R # defines a vector instead of computing a sum later
    #R is the density of links (because it works on j spatial position) and not time)


def compute_R_power_law(length: int, exponent: float = 1.5) -> np.ndarray:
    """
    Compute R_j coefficients with power law decay.
    R_j = j^(-exponent)
    """
    R = np.zeros(length)
    R[0] = 0  # R_0 is not used
    for j in range(1, length):
        R[j] = j ** (-exponent)
    return R


# ============================================================================
# Main program
# ============================================================================

def main():
    """
    Main program for solving the Volterra equation.
    """
    print("=" * 70)
    print("Volterra Equation Solver")
    print("Equation: h * sum_{j>=1} psi(Z^n - Z^{n-j}) * R_j = f^n")
    print("=" * 70)
    
    # ========================================================================
    # Parameters (modify these before running)
    # ========================================================================
    
    # Dimension (1 or 2)
    d = 1
    
    # Power alpha for psi(u) = u^alpha
    alpha = 2
    
    # Time parameters
    h = 0.01        # Time step
    N = 50          # Number of initial points
    M = 50        # Number of points to compute

    # Notre implémentation
    delta_x = 0.01
    N_x = 60
    
    # Forcing function choice: 'constant', 'oscillating', or 'random'
    forcing_type = 'oscillating'
    
    # Parameters for forcing function
    f_amplitude = 1.0
    f_frequency = 1.0
    f_seed = 42
    
    # R_j decay type: 'exponential' or 'power_law'
    R_decay_type = 'power_law'
    decay_rate = 0.5      # For exponential decay
    power_exponent = 1.5  # For power law decay
    
    # Initial condition amplitude
    z_initial_amplitude = 0.1
    
    # Plot and save options
    do_plot = True
    save_filename = "outputs/residence_time"
    
    # ========================================================================
    # Setup based on parameters
    # ========================================================================
    
    print(f"\nParameters:")
    print(f"  Dimension: d = {d}")
    print(f"  Power: alpha = {alpha}")
    print(f"  Time step: h = {h}")
    print(f"  Initial points: N = {N}")
    print(f"  Compute points: M = {M}")
    
    # Setup forcing function
    if forcing_type == 'constant':
        f_func = f_constant(d, value=f_amplitude)
        print(f"  Forcing: constant, f = {f_amplitude}")
    elif forcing_type == 'oscillating':
        f_func = f_oscillating(d, f_amplitude, f_frequency)
        print(f"  Forcing: oscillating, amplitude = {f_amplitude}, frequency = {f_frequency}")
    elif forcing_type == 'random':
        f_func = f_random(d, f_amplitude, f_seed)
        print(f"  Forcing: random, amplitude = {f_amplitude}, seed = {f_seed}")
    else:
        print(f"  Warning: Unknown forcing type '{forcing_type}', using constant")
        f_func = f_constant(d, value=1.0)
    
    # Setup R_j coefficients
    if R_decay_type == 'exponential':
        R = compute_R_exponential(M + N + 10, decay_rate)
        print(f"  R_j: exponential decay, rate = {decay_rate}")
    elif R_decay_type == 'power_law':
        R = compute_R_power_law(M + N + 10, power_exponent)
        print(f"  R_j: power law decay, exponent = {power_exponent}")
    else:
        print(f"  Warning: Unknown R decay type '{R_decay_type}', using exponential")
        R = compute_R_exponential(M + N + 10, 0.5)
    
    # Setup initial conditions
    z_initial = z_initial_default(d, amplitude=z_initial_amplitude)
    
    # ========================================================================
    # Solve
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("Solving...")
    print("=" * 70)
    
    solver = VolterraSolver(
        h=h,
        delta_x = delta_x,
        N_x = N_x,
        N=N,
        M=M,
        d=d,
        alpha=alpha,
        R=R,
        f_func=f_func,
        z_initial=z_initial
    )
    
    t, Z = solver.solve()
    
    print("\nSolution complete!")
    print(f"Final value Z^{M}: {Z[-1]}")
    
    # ========================================================================
    # Plot
    # ========================================================================
    
    if do_plot:
        print("\nPlotting solution...")
        solver.plot_solution(save_filename, t, Z)
    
    # Save results
    np.savez(f"{save_filename}.npz", t=t, Z=Z, alpha=alpha, d=d, h=h, N=N, M=M)
    print(f"Results saved to {save_filename}.npz")
    



if __name__ == "__main__":
    main()
