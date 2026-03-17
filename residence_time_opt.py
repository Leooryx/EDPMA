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
        decay_rate: float,
        f_func: Callable[[float], np.ndarray],
        z_initial: Callable[[float], np.ndarray],
        stiffness: int
    ):
        """
        Parameters:
        -----------
        h : float
            Time step # delta_t
        delta_x : float
            Space step
        N_x : int
            Max position in space
        N : int
            Number of initial points (for n in [-N, 0)) # points for past history
        M : int
            Number of points to compute (for n in [0, M])
        d : int
            Dimension of Z (1 or 2)
        alpha : float
            Power in psi(u) = u^alpha
        decay_rate : float
            decay rate of local time weight
        f_func : callable
            Function f(t) returning d-dimensional vector
        z_initial : callable
            Initial condition function z_p(t) for t < 0
        stiffness : int
            stiffness constant
        """
        self.h = h
        self.delta_x = delta_x
        self.N_x = N_x
        self.N = N
        self.M = M
        self.d = d
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.f_func = f_func
        self.z_initial = z_initial
        self.stiffness = stiffness
        
        self.X_grid = np.arange(self.N_x, step=self.delta_x) 
        self.A_minus_N = np.zeros( int(self.N_x / self.delta_x)) 

        self.Z = np.zeros((N + M + 1, d)) 

        # Initialize with initial conditions
        for n in range(-N, 0): 
            self.Z[n + N] = self.z_initial(n * h) 
            #particle goes between 0 and 0.09990005 at initial time

        def initial_local_time(Z):
            qt = 1-self.decay_rate*self.h
            for n in range(-N, 0):
                if n == -N:
                    A_initial = self.A_minus_N.copy()
                else:
                    A_initial *= qt
                    k_idx = (self.Z[n+N] / self.delta_x).astype(int)
                    A_initial[k_idx] += self.h
            return A_initial

        self.A_initial = initial_local_time(self.Z)

        
    
    def psi(self, u: np.ndarray) -> np.ndarray:
        """
        Nonlinearity psi(u) = u^alpha (component-wise).
        """
        return self.stiffness * np.abs(u) ** self.alpha

    
    def local_time(self, n):
        #"""A_x^n = \Delta t \delta_{x=z^{n-1}} + (1 - \lambda \Delta t)A_x^{n-1}"""
        
        #qt = np.exp(-self.decay_rate * self.h) 
        qt = 1-self.decay_rate*self.h
        
        if n==0:
            self.A = self.A_initial.copy() 
        else:
            self.A *= qt
            z_n = self.Z[n + self.N - 1]
            k_idx = (z_n / self.delta_x).astype(int)
            self.A[k_idx] += self.h
        #no return we just update the list


    def residual(self, z_n: np.ndarray, n: int) -> np.ndarray:
        
        res = self.psi(z_n - self.X_grid) * self.A 
        res = np.sum(res) #* self.delta_x
        res -= self.f_func(n * self.h)
        
        return res
    

    def solve_step(self, n: int) -> np.ndarray:
        """
        Solve for Z^n using nonlinear solver.
        """
        # local time update
        self.local_time(n)
        
        # Initial guess: use previous value
        Z_guess = self.Z[n + self.N - 1] if n > 0 else self.Z[self.N - 1]

        solution = fsolve(
            lambda z: self.residual(z, n),
            Z_guess,
            full_output=False
        )
        
        return solution
    
    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
    
        for n in range(self.M +1):
            self.Z[n + self.N] = self.solve_step(n)
        t = np.array([n * self.h for n in range(-self.N, self.M + 1)])
        
        return t, self.Z





# ============================================================================
# Main program for tests
# ============================================================================

def main():
    """
    Main program for solving the Volterra equation.
    """
    
    
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
    M = 250      # Number of points to compute

    z_initial_amplitude = 0.1

    # Space steps and interval
    delta_x = 0.01
    N_x = 500

    lamb = 4
    decay_rate = 1/lamb
    f_amplitude =  1    #50 #c in our math. 
    stiffness = 1  #10000

    asymptotic_speed = (f_amplitude * decay_rate**3 / (2*stiffness))**(1/2)

    file_name = "outputs/residence_time_opt"
    

    
    def f_constant(d: int, value: float = 1.0) -> Callable[[float], np.ndarray]:
        """Constant forcing function."""
        return lambda t: value * np.ones(d)
    f_func = f_constant(d, value=f_amplitude)
    

    def z_initial_default(d: int, amplitude: float = 0.1) -> Callable[[float], np.ndarray]:
        """Default initial condition: exponential decay as t -> -infinity."""
        return lambda t: amplitude * np.exp(t / 10) * np.ones(d) # the past history, as a function of the time
    z_initial = z_initial_default(d, amplitude=z_initial_amplitude)
    


    # ========================================================================
    # Solve
    # ========================================================================
    
    solver = VolterraSolver(
        h=h,
        delta_x = delta_x,
        N_x = N_x,
        N=N,
        M=M,
        d=d,
        alpha=alpha,
        decay_rate=decay_rate,
        f_func=f_func,
        z_initial=z_initial,
        stiffness=stiffness
    )
    
    t, Z = solver.solve()
    speed = np.gradient(Z, h, axis=0)
    acceleration = np.gradient(speed, h, axis=0)
    #remove past history for better plotting
    t, Z, speed, acceleration = t[N+1:], Z[N+1:], speed[N+1:], acceleration[N+1:]
    

    # Plot    
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    axes[0].plot(t, Z, 'b-', linewidth=1.5, label=f'$Z(t)$')
    axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='$t=0$')
    axes[0].set_xlabel('$t$')
    axes[0].set_ylabel('$Z$')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Leukocyte position")

    axes[1].plot(t, speed, 'b-', linewidth=1.5, label=f'$\\nabla Z(t)$')
    axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='$t=0$')
    axes[1].axhline(y=asymptotic_speed, color='green', linestyle='--', alpha=0.5, label=f'Asymptotic speed: {round(asymptotic_speed, 3)}')
    axes[1].set_xlabel('$t$')
    axes[1].set_ylabel(f'$\\nabla Z$')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Leukocyte speed")

    axes[2].plot(t[1:], acceleration[1:], 'b-', linewidth=1.5, label=f'$\\nabla^2 Z(t)$')
    axes[2].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='$t=0$')
    axes[2].set_xlabel('$t$')
    axes[2].set_ylabel(f'$\\nabla^2 Z$')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title("Leukocyte acceleration")
 
    plt.suptitle(f'Solution for $\\psi(u) = u^{{{alpha}}}$ for stiffness={stiffness}, force={f_amplitude}')
    plt.tight_layout()
    plt.savefig(f'/home/onyxia/work/EDPMA/{file_name}.png')
    plt.show()
    



if __name__ == "__main__":
    main()