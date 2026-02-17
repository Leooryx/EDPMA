"""
Animation of white blood cell adhesion to blood vessel wall.
Uses the Volterra solver to compute cell dynamics under blood flow.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
from tqdm import tqdm

def create_animation(t, Z, h, N):
   
    # Setup figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.3)
    
    # Get data range
    t_plot = t[0:]
    Z_plot = Z[0:, :] # we remove initialisation points by starting with N
    x_data = Z_plot[:, 0]
    y_data = Z_plot[:, 1]

    pbar = tqdm(range(len(t_plot)), desc="Generating Animation")
    
    # --- Top panel: Vessel Visualization (The "Camera" View) ---
    ax1.set_ylim(-1, 400)
    ax1.set_aspect('equal')
    ax1.set_xlabel('Position (μm)', fontsize=10)
    ax1.set_ylabel('Height (μm)', fontsize=10)
    ax1.set_title('vessel', fontsize=12, fontweight='bold')
    
    # Vessel wall
    vessel_limit = max(x_data) + 10
    vessel_bottom = Rectangle((-5, -1), vessel_limit + 5, 0.3, color='#8B4513', alpha=0.7)
    ax1.add_patch(vessel_bottom)
    
    # White blood cell
    cell_radius = 0.5
    cell = Circle((x_data[0], y_data[0]), cell_radius, facecolor='white', 
                  edgecolor='blue', linewidth=2, zorder=10)
    ax1.add_patch(cell)
    
    # Trajectory trail for top plot
    trail, = ax1.plot([], [], 'b--', alpha=0.3, linewidth=1)
    
    # Texts
    time_text = ax1.text(0.02, 0.92, '', transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    state_text = ax1.text(0.75, 0.92, '', transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    # --- Bottom panel: 2D Path Plot (Y vs X) ---
    ax2.set_xlim(min(x_data) - 1, max(x_data) + 1)
    ax2.set_ylim(-0.5, max(y_data) + 1)
    ax2.set_xlabel('Horizontal position x (μm)', fontsize=10)
    ax2.set_ylabel('Vertical height y (μm)', fontsize=10)
    ax2.set_title('2D trajectory path', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # The "Path" line and the "Current Position" dot
    path_line, = ax2.plot([], [], 'g-', linewidth=1.5, alpha=0.7, label='Cell Path')
    path_dot, = ax2.plot([], [], 'ro', markersize=6, label='Current Position')
    ax2.axhline(y=0.3, color='brown', linestyle='--', alpha=0.4, label='Vessel Wall Surface')
    ax2.legend(loc='upper right')

    def init():
        trail.set_data([], [])
        path_line.set_data([], [])
        path_dot.set_data([], [])
        time_text.set_text('')
        state_text.set_text('')
        return trail, cell, path_line, path_dot, time_text, state_text

    def animate(frame):
        curr_x = x_data[frame]
        curr_y = y_data[frame]
        
        # 1. Update Cell and "Camera" (Top Plot)
        cell.center = (curr_x, curr_y)
        # Follow the cell: set x-limits to be centered around the cell
        ax1.set_xlim(min(x_data) - 2, max(x_data) + 2)
        
        # Update trail
        trail_start = max(0, frame - 100)
        trail.set_data(x_data[trail_start:frame], y_data[trail_start:frame])
        
        # 2. Update Path (Bottom Plot)
        path_line.set_data(x_data[:frame], y_data[:frame])
        path_dot.set_data([curr_x], [curr_y])
        
        # 3. Update Status and Colors
        time_text.set_text(f'Time: {t_plot[frame]:.2f} s')
        
        if curr_y > 1.5:
            state = "Free Flow"
            cell.set_facecolor('#008000') # Green
        elif curr_y > 0.6:
            state = "Rolling/Tethering"
            cell.set_facecolor('#FFF9C4') # Light Yellow
        else:
            state = "Adhered"
            cell.set_facecolor('#FFCDD2') # Light Red
        
        state_text.set_text(f'State: {state}')
        
        return trail, cell, path_line, path_dot, time_text, state_text

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=pbar, interval=30, blit=True
    )

    pbar.close()
    
    return fig, anim