import scipy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.fft import fft, fftfreq
from scipy.signal import detrend

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

def get_main_frequency(signal, sample_rate):
    # on retire les tendances (par exemple la décroissance de la vitesse) pour nettoyer le signal
    signal_clean = detrend(signal.flatten())

    n = len(signal_clean)
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

taille_grilles = 100
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
        #on retire le passé qui cause une forte discontinuité
        t, Z, Z_speed = t[N+1:], Z[N+1:], Z_speed[N+1:]
        sample_rate = len(Z_speed)/h 
        frec = get_main_frequency(signal=Z_speed, sample_rate=sample_rate)
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