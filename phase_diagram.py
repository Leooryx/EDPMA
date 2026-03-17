import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from residence_time_opt import VolterraSolver

# ========================================================================
# Parameters 
# ========================================================================
    
# Dimension (1 or 2)
d = 1

# Power alpha for psi(u) = u^alpha
alpha = 2

# Time parameters
h = 0.01        # Time step
N = 50          # Number of initial points
M = 500        # Number of points to compute

z_initial_amplitude = 0.1

# Space steps and interval
delta_x = 0.01
N_x = 600

decay_rate = 0.5
f_amplitude = 9.0
stiffness = 150000

def f_constant(d, value):
    return lambda t: value * np.ones(d)
f_func = f_constant(d, value=f_amplitude)


def z_initial_default(d, amplitude):
    return lambda t: amplitude * np.exp(t / 10) * np.ones(d) # the past history, as a function of the time
z_initial = z_initial_default(d, amplitude=z_initial_amplitude)


# ========================================================================
# Fonctions utiles
# ========================================================================

def get_first_WL(signal, dt):
    L = len(signal)
    first_jump_index = 0
    second_jump_index = 0
    for i in range(2, L):
        instant_index_variation = signal[i]-signal[i-1]
        delayed_index_variation = signal[i]-signal[i-2]
        if instant_index_variation >= 7*delayed_index_variation :
            first_jump_index = i
            break
    for j in range(first_jump_index+2, L):
        instant_index_variation = signal[j]-signal[j-1]
        delayed_index_variation = signal[j]-signal[j-2]
        if instant_index_variation >= 10*delayed_index_variation :
            second_jump_index = j
            break
    delta_index = second_jump_index-first_jump_index
    WL = delta_index*dt
    return WL

#=================================================
#code pour génération des données
#=================================================

taille_grilles = 5
phase_array = []

grille_f_amplitude = np.linspace(0 , 10 , taille_grilles)
grille_raideur = np.linspace(0 , 150000 , taille_grilles)

for i in tqdm(range(taille_grilles), desc="Steps"):
    phase_array.append([])
    for j in range(taille_grilles):
        f_func = f_constant(d, value=grille_f_amplitude[i])
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
            stiffness=grille_raideur[j]
        )
        
        t, Z = solver.solve()
        Z_speed = np.gradient(Z, axis=0)
        Z_accel = np.gradient(Z_speed, axis=0)
        #on retire le passé qui cause une forte discontinuité
        t, Z, Z_speed, Z_accel = t[N+1:], Z[N+1:], Z_speed[N+1:], Z_accel[N+1:]
        first_WL = get_first_WL(signal=Z_accel, dt=h)
        phase_array[-1].append(first_WL)


phase_array = np.array(phase_array)


fig, ax = plt.subplots()
im = ax.imshow(
    phase_array,
    extent=[grille_f_amplitude.min(), grille_f_amplitude.max(),
            grille_raideur.min(), grille_raideur.max()],
    origin="lower",
    aspect="auto",
)
fig.colorbar(im, ax=ax)
plt.show()
plt.savefig(f'phase_diagram.png')