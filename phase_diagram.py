import scipy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.fft import fft, fftfreq

from residence_time_opt import VolterraSolver, f_constant, z_initial_default, compute_R_exponential

# ========================================================================
# Parameters (pas touche à ceux-ci)
# ========================================================================
    
# Dimension (1 or 2)
d = 1
    
# Power alpha for psi(u) = u^alpha
alpha = 2
    
# Time parameters
h = 0.01        # Time step
N = 50          # Number of initial points
M = 1000        # Number of points to compute

# Notre implémentation
delta_x = 0.01
N_x = 600
    
# Forcing function choice: 'constant', 'oscillating', or 'random'
forcing_type = 'constant'
    
# Parameters for forcing function
#f_amplitude = 1.0
f_frequency = 1.0
f_seed = 42
    
# R_j decay type: 'exponential' or 'power_law'
R_decay_type = 'exponential'
#decay_rate = 0.5      # For exponential decay
power_exponent = 1.5  # For power law decay
    
# Initial condition amplitude
z_initial_amplitude = 0.1

#fonction z_init
z_initial = z_initial_default(d, amplitude=z_initial_amplitude)

#=================================================
#parametres à faire varier pour diagramme de phase
#=================================================

#psi param
raideur = 500000
#R_j decay param
decay_rate = 0.5
#f_constant force value
f_amplitude = 1.0

#pas touche, mais varie avec ce qu'il y a au dessus
#setup pour les coeff de R_j
R = compute_R_exponential(M + N + 10, decay_rate)


#=================================================
#fonctions utiles
#=================================================

def get_main_frequency(signal, sample_rate):
    n = len(signal)
    
    # Compute FFT and frequencies
    fft_vals = fft(signal)
    freqs    = fftfreq(n, d=1/sample_rate)  # d = time spacing between samples
    
    # Keep only positive frequencies and their magnitudes
    pos_mask    = freqs > 0
    freqs_pos   = freqs[pos_mask]
    magnitudes  = np.abs(fft_vals[pos_mask])
    
    # Pick the frequency with the highest magnitude
    main_freq = freqs_pos[np.argmax(magnitudes)]
    return main_freq

#=================================================
#code pour génération des données
#=================================================

taille_grilles = 10
phase_array = []

grille_f_amplitude = np.linspace(0 , 10 , taille_grilles)
grille_raideur = np.linspace(0 , 10**6/2 , taille_grilles)

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
            R=R,
            f_func=f_func,
            z_initial=z_initial,
            raideur = grille_raideur[j])
        
        t, Z = solver.solve()
        Z_speed = np.gradient(Z, h, axis=0)
        sample_rate = len(Z_speed)/(M*delta_x)
        frec = get_main_frequency(signal= Z_speed, sample_rate=sample_rate)
        phase_array[-1].append(frec)


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
plt.savefig(f'/home/onyxia/work/EDPMA/phase_diagram.png')