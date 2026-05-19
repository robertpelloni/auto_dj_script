# 📐 Technical Design & Mathematical Foundations (v5.0.0)

## 1. Genre-Aware Heuristics
The classification engine uses a combination of **Spectral Centroid** ($\mu_c$) and **Spectral Rolloff** ($\mu_r$) to determine stylistic archetypes:
- **High-Energy**: $\mu_c > 3000\text{Hz}$ or $\mu_r > 6000\text{Hz}$.
- **Techno**: $1500\text{Hz} < \mu_c \leq 3000\text{Hz}$.
- **Ambient**: $\mu_c \leq 1500\text{Hz}$.

## 2. Dynamic Temporal Warping
Tempo gradient implementation:
$$\text{BPM}(t) = \text{BPM}_{\text{start}} + \left( \frac{\text{BPM}_{\text{end}} - \text{BPM}_{\text{start}}}{D_{\text{set}}} \right) \cdot t$$
To maintain fidelity, the track is segmented into $\Delta t = 1\text{s}$ chunks. Each chunk $i$ is processed by a phase vocoder with a discrete rate $R_i = \frac{\text{BPM}(t_i)}{\text{BPM}_{\text{native}}}$.

## 3. Global Optimization
Transition Score Function:
$$S = W_h \cdot \text{KeyMatch} + W_s \cdot \text{SyncPotential} + W_e \cdot \text{EnergyFlow} + W_g \cdot \text{GenreMatch}$$
The simulated annealing cooling schedule uses $T_{n+1} = T_n \cdot 0.99$, exploring the state space to minimize $E = -S$.
