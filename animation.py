"""
Animation of white blood cell adhesion to blood vessel wall.
Uses the Volterra solver to compute cell dynamics under blood flow.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Ellipse
from scipy.optimize import fsolve
from typing import Callable, Tuple


class VolterraSolver:
    """
    Solves the discrete Volterra-type equation with power nonlinearity.
    """
    
    def __init__(
        self,
        h: float,
        N: int,
        M: int,
        d: int,
        alpha: float,
        R: np.ndarray,
        f_func: Callable[[float], np.ndarray],
        z_initial: Callable[[float], np.ndarray]
    ):
        self.h = h
        self.N = N
        self.M = M
        self.d = d
        self.alpha = alpha
        self.R = R
        self.f_func = f_func
        self.z_initial = z_initial
        
        # Storage for solution
        self.Z = np.zeros((N + M + 1, d))
        
        # Initialize with initial conditions
        for n in range(-N, 0):
            self.Z[n + N] = self.z_initial(n * h)
    
    def psi(self, u: np.ndarray) -> np.ndarray:
        """Nonlinearity psi(u) = u^alpha (component-wise)."""
        if self.alpha == int(self.alpha) and int(self.alpha) % 2 == 1:
            return np.sign(u) * np.abs(u) ** self.alpha
        else:
            return np.abs(u) ** self.alpha
    
    def residual(self, Z_n: np.ndarray, n: int) -> np.ndarray:
        """Compute residual: h * sum_{j>=1} psi(Z^n - Z^{n-j}) * R_j - f^n"""
        res = np.zeros(self.d)
        
        for j in range(1, n + self.N + 1):
            if j < len(self.R):
                idx_prev = n + self.N - j
                Z_diff = Z_n - self.Z[idx_prev]
                res += self.h * self.psi(Z_diff) * self.R[j]
        
        res -= self.f_func(n * self.h)
        
        return res
    
    def solve_step(self, n: int) -> np.ndarray:
        """Solve for Z^n using nonlinear solver."""
        Z_guess = self.Z[n + self.N - 1] if n > 0 else self.Z[self.N - 1]
        
        solution = fsolve(
            lambda z: self.residual(z, n),
            Z_guess,
            full_output=False
        )
        
        return solution
    
    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the equation for n = 0, 1, ..., M."""
        for n in range(0, self.M + 1):
            self.Z[n + self.N] = self.solve_step(n)
            
            if (n + 1) % 100 == 0:
                print(f"Solved up to n = {n}")
        
        t = np.array([n * self.h for n in range(-self.N, self.M + 1)])
        
        return t, self.Z


def compute_R_power_law(length: int, exponent: float = 1.5) -> np.ndarray:
    """Compute R_j coefficients with power law decay."""
    R = np.zeros(length)
    R[0] = 0
    for j in range(1, length):
        R[j] = j ** (-exponent)
    return R


def setup_cell_dynamics():
    """
    Setup Volterra solver for white blood cell dynamics.
    
    The 2D system represents:
    - Z[0]: horizontal position (along blood flow)
    - Z[1]: vertical position (distance from vessel wall)
    """
    
    # Parameters
    d = 2  # 2D motion
    alpha = 1.5  # Nonlinearity parameter
    h = 0.05  # Time step
    N = 20  # Initial history points
    M = 300  # Points to compute
    
    # R_j: represents adhesion molecule density (power law decay)
    R = compute_R_power_law(M + N + 10, exponent=1.8)
    
    # Forcing function: blood flow pushes cell horizontally,
    # adhesion molecules pull vertically toward wall
    def f_blood_flow(t):
        # Horizontal: constant flow with slight pulsation
        flow_x = 3.0 + 0.5 * np.sin(2 * np.pi * 0.3 * t)
        
        # Vertical: adhesion force increases over time (rolling -> firm adhesion)
        # Negative means toward the wall (bottom)
        adhesion_y = -0.5 * (1 - np.exp(-0.3 * t))
        
        return np.array([flow_x, adhesion_y])
    
    # Initial condition: cell starts slightly above wall, moving with flow
    def z_initial_cell(t):
        return np.array([
            2.0 + 0.1 * np.exp(t / 5),  # x position
            3.0 + 0.1 * np.exp(t / 5)   # y position (above wall)
        ])
    
    # Create solver
    solver = VolterraSolver(
        h=h, N=N, M=M, d=d, alpha=alpha,
        R=R, f_func=f_blood_flow, z_initial=z_initial_cell
    )
    
    print("Computing white blood cell dynamics...")
    t, Z = solver.solve()
    print("Dynamics computed!")
    
    return t, Z, h, N


def create_animation(t, Z, h, N):
    """Create animation of white blood cell adhesion."""
    
    # Setup figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Get data range
    t_plot = t[N:]  # Only forward time
    Z_plot = Z[N:, :]
    
    x_data = Z_plot[:, 0]
    y_data = Z_plot[:, 1]
    
    # --- Top panel: Blood vessel visualization ---
    ax1.set_xlim(-1, max(x_data) + 3)
    ax1.set_ylim(-1, 8)
    ax1.set_aspect('equal')
    ax1.set_xlabel('Position (μm)', fontsize=12)
    ax1.set_ylabel('Height (μm)', fontsize=12)
    ax1.set_title('White Blood Cell Adhesion to Vessel Wall', fontsize=14, fontweight='bold')
    
    # Draw blood vessel walls
    vessel_bottom = Rectangle((-1, -1), max(x_data) + 4, 0.3, 
                             color='#8B4513', alpha=0.7, label='Vessel wall')
    ax1.add_patch(vessel_bottom)
    
    vessel_top = Rectangle((-1, 7), max(x_data) + 4, 0.3, 
                          color='#8B4513', alpha=0.7)
    ax1.add_patch(vessel_top)
    
    # Draw endothelial cells on vessel wall
    for i in range(0, int(max(x_data)) + 4, 2):
        cell_patch = Rectangle((i, -0.7), 1.8, 0.4, 
                              color='#CD853F', alpha=0.5, linewidth=0.5, edgecolor='brown')
        ax1.add_patch(cell_patch)
    
    # Blood flow direction arrow
    ax1.annotate('', xy=(max(x_data) + 2, 6.5), xytext=(max(x_data) - 1, 6.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='red', alpha=0.6))
    ax1.text(max(x_data) + 0.5, 6.8, 'Blood Flow', fontsize=10, color='red', 
            ha='center', fontweight='bold')
    
    # White blood cell (will be updated)
    cell = Circle((x_data[0], y_data[0]), 0.4, color='white', 
                 edgecolor='blue', linewidth=2, zorder=10)
    ax1.add_patch(cell)
    
    # Cell trajectory trail
    trail, = ax1.plot([], [], 'b--', alpha=0.4, linewidth=1, label='Cell trajectory')
    
    # Add adhesion molecules visualization
    adhesion_points = []
    
    # Time text
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, 
                        fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # State text
    state_text = ax1.text(0.02, 0.85, '', transform=ax1.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # --- Bottom panel: Position traces ---
    ax2.set_xlim(t_plot[0], t_plot[-1])
    ax2.set_ylim(min(min(x_data), min(y_data)) - 1, max(max(x_data), max(y_data)) + 1)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Position (μm)', fontsize=12)
    ax2.set_title('Cell Position Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    line_x, = ax2.plot([], [], 'r-', linewidth=2, label='x (horizontal)')
    line_y, = ax2.plot([], [], 'b-', linewidth=2, label='y (vertical)')
    time_marker_x, = ax2.plot([], [], 'ro', markersize=8)
    time_marker_y, = ax2.plot([], [], 'bo', markersize=8)
    
    ax2.axhline(y=0, color='brown', linestyle='--', alpha=0.5, linewidth=1.5, label='Vessel wall')
    ax2.legend(loc='upper right', fontsize=10)
    
    def init():
        """Initialize animation."""
        trail.set_data([], [])
        line_x.set_data([], [])
        line_y.set_data([], [])
        time_marker_x.set_data([], [])
        time_marker_y.set_data([], [])
        return trail, cell, line_x, line_y, time_marker_x, time_marker_y, time_text, state_text
    
    def animate(frame):
        """Update animation frame."""
        # Update cell position
        cell.center = (x_data[frame], y_data[frame])
        
        # Update trajectory trail
        trail_length = min(50, frame)
        trail.set_data(x_data[max(0, frame-trail_length):frame+1], 
                      y_data[max(0, frame-trail_length):frame+1])
        
        # Update position plots
        line_x.set_data(t_plot[:frame+1], x_data[:frame+1])
        line_y.set_data(t_plot[:frame+1], y_data[:frame+1])
        time_marker_x.set_data([t_plot[frame]], [x_data[frame]])
        time_marker_y.set_data([t_plot[frame]], [y_data[frame]])
        
        # Update time text
        time_text.set_text(f'Time: {t_plot[frame]:.2f} s')
        
        # Determine cell state based on height
        height = y_data[frame]
        if height > 2.0:
            state = "Free rolling"
            cell.set_facecolor('white')
            cell.set_edgecolor('blue')
        elif height > 0.8:
            state = "Tethering"
            cell.set_facecolor('lightyellow')
            cell.set_edgecolor('orange')
        else:
            state = "Firm adhesion"
            cell.set_facecolor('lightcoral')
            cell.set_edgecolor('red')
        
        state_text.set_text(f'State: {state}\nHeight: {height:.2f} μm')
        
        # Add adhesion molecules when cell gets close to wall
        if height < 1.5 and frame % 3 == 0:
            x_pos = x_data[frame]
            # Add small markers representing adhesion molecules
            if len(adhesion_points) < 30:  # Limit number
                marker = ax1.plot([x_pos], [0.3], 'g*', markersize=6, alpha=0.6)[0]
                adhesion_points.append(marker)
        
        return (trail, cell, line_x, line_y, time_marker_x, time_marker_y, 
                time_text, state_text, *adhesion_points)
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(t_plot), interval=50, blit=True, repeat=True
    )
    
    return fig, anim


def main():
    """Main function to run the animation."""
    print("=" * 70)
    print("White Blood Cell Adhesion Animation")
    print("Using Volterra Equation for Cell Dynamics")
    print("=" * 70)
    
    # Compute cell dynamics
    t, Z, h, N = setup_cell_dynamics()
    
    # Create and display animation
    print("\nCreating animation...")
    fig, anim = create_animation(t, Z, h, N)
    
    # Save animation
    '''print("Saving animation...")
    anim.save('white_cell_adhesion.mp4', 
              writer='ffmpeg', fps=20, dpi=100)
    print("Animation saved to: white_cell_adhesion.mp4")'''
    
    # Also save as GIF (smaller file)
    print("Saving as GIF...")
    anim.save('white_cell_adhesion.gif', 
              writer='pillow', fps=20)
    print("GIF saved to: white_cell_adhesion.gif")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()