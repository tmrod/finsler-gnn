import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
from abc import ABC, abstractmethod
from typing import Callable

import optax

import diffrax
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController, RecursiveCheckpointAdjoint

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

key = jr.PRNGKey(42) # RNG
    
d = 2 # sizes of graphs for training and testing

# ==========================================
# Representation of Finsler Metrics
# ==========================================

class MetricTransform(ABC, eqx.Module):
    @abstractmethod
    def E(self, xi):
        pass

    @abstractmethod
    def J(self, xi):
        pass

class RandersMetric(MetricTransform):
    b: jax.Array

    def __init__(self, b):
        self.b = b

    def E(self, xi):
        norm = jnp.sqrt(jnp.sum(xi**2) + 1e-6)
        return 0.5 * (norm + jnp.inner(self.b, xi))**2

    def J(self, xi):
        return jax.grad(self.E)(xi)

class SymmetricNeuralMetric(MetricTransform):
    W: jax.Array
    activ: Callable

    def __init__(self, input_dim, hidden_dim, activ=jax.nn.relu, key=jr.PRNGKey(0)):
        self.W = jax.random.normal(key, (hidden_dim, input_dim)) / jnp.sqrt(input_dim * hidden_dim)
        self.activ = activ

    def J(self, xi):
        return self.W.T @ self.activ(self.W @ xi)

    def E(self, xi):
        return 0.5 * jnp.inner(xi, self.J(xi))

# ==========================================
# Graph Data Structure
# ==========================================

class PointGraph():
    dx: jax.Array
    C_inv: jax.Array
    K: jax.Array
    n: int
    d: int
    D: int
    eps: float

    def __init__(self, dx, d, eps, eps_pinv=1e-5):
        self.dx = dx
        self.n = dx.shape[0]
        self.D = dx.shape[2]
        self.d = d
        self.eps = eps

        sq_dist = jnp.sum(self.dx**2, axis=-1)
        self.K = jnp.exp(-sq_dist / (2 * self.eps**2))
        
        dx_outer = dx[:, :, :, None] * dx[:, :, None, :]
        C = jnp.sum(self.K[:, :, None, None] * dx_outer, axis=1)
        C /= (self.n * self.eps**(self.d+2))
        # rank-d pseudoinverse
        Lam, V = jnp.linalg.eigh(C)
        Lam_inv = (1./Lam).at[:,0].set(0)
        self.C_inv = jnp.einsum('nij,nj,nkj->nik', V, Lam_inv, V)

    def grad(self, f):
        df = f[:, None] - f[None, :]
        b_vec = jnp.sum(self.K[:, :, None] * df[:, :, None] * self.dx, axis=1)
        b_vec /= (self.n * self.eps**(self.d+2))
        return jnp.einsum('nij,nj->ni', self.C_inv, b_vec)
        
    def cograd(self, g):
        h = jnp.einsum('nij,nj->ni', self.C_inv, g)
        h_sum = h[:, None, :] + h[None, :, :]
        h_proj = jnp.sum(h_sum * self.dx, axis=-1)
        return jnp.sum(self.K * h_proj, axis=1) / (self.n * self.eps**(self.d+2))

    def laplacian(self, f, J=jax.nn.identity):
        return self.cograd(jax.vmap(J)(self.grad(f)))

# ==========================================
# Synthetic Data Generation
# ==========================================

R, r = 2.0, 1.0  # Major and minor radii

def sample_torus(n, key):
    key_u, key_v, key_p = jr.split(key, 3)
    oversample = n * 4 
    
    u = jr.uniform(key_u, shape=(oversample,), minval=0.0, maxval=2*jnp.pi)
    v_cand = jr.uniform(key_v, shape=(oversample,), minval=0.0, maxval=2*jnp.pi)
    
    p_cand = jr.uniform(key_p, shape=(oversample,), minval=0.0, maxval=R+r)
    accept_mask = p_cand <= (R + r * jnp.cos(v_cand))
    
    u_acc = u[accept_mask][:n]
    v_acc = v_cand[accept_mask][:n]
    
    x = (R + r * jnp.cos(v_acc)) * jnp.cos(u_acc)
    y = (R + r * jnp.cos(v_acc)) * jnp.sin(u_acc)
    z = r * jnp.sin(v_acc)
    
    return jnp.stack([x, y, z], axis=-1), u_acc, v_acc

def f_ambient(pt):
    x, y, z = pt[0], pt[1], pt[2]
    return jnp.cos(x) + jnp.cos(y) + z

def get_tangent_projector(pt):
    x, y, z = pt[0], pt[1], pt[2]
    
    u = jnp.arctan2(y, x)
    
    nx = jnp.cos(u) * (jnp.sqrt(x**2 + y**2) - R)
    ny = jnp.sin(u) * (jnp.sqrt(x**2 + y**2) - R)
    nz = z
    
    n_vec = jnp.array([nx, ny, nz])
    n_vec = n_vec / jnp.linalg.norm(n_vec)
    
    return jnp.eye(3) - jnp.outer(n_vec, n_vec)

def J_true(xi):
    # J for F(xi)=\|xi\|_3
    return xi*jnp.abs(xi) / (jnp.sum(jnp.abs(xi)**3) + 1e-9)**(1./3.)

def exact_manifold_laplacian(pt):
    P = get_tangent_projector(pt)
    
    ambient_grad_fn = jax.grad(f_ambient)
    
    def flux_fn(y):
        grad_y = ambient_grad_fn(y)
        P_y = get_tangent_projector(y)
        tangent_grad = jnp.matmul(P_y, grad_y)
        flux = J_true(tangent_grad)
        return jnp.matmul(P_y, flux)

    jacobian = jax.jacfwd(flux_fn)(pt)
    projected_jacobian = jnp.matmul(P, jacobian)
    
    return jnp.trace(projected_jacobian)

vmap_exact_lap = jax.vmap(exact_manifold_laplacian)
vmap_f = jax.vmap(f_ambient)



n_values = jnp.array([250,  500,  750,  1000, 1250,
                      1500, 1750, 2000, 2250, 2500,
                      2750, 3000, 3250, 3500, 3750,
                      4000, 4250, 4500, 4750, 5000,
                      5250, 5500, 5750, 6000, 6250,
                      6500, 6750, 7000, 7250, 7500,
                      7750, 8000, 8250, 8500, 8750,
                      9000, 9250, 9500])
eps_values = jnp.linspace(0.05, 0.6, n_values.shape[0], endpoint=True)
error_matrix = jnp.zeros((len(eps_values), len(n_values)))

for j, n in enumerate(n_values):
    key, subkey = jax.random.split(key)
    X, _, _ = sample_torus(n, subkey)
    f_val = vmap_f(X)
    lap_true = vmap_exact_lap(X)
    dx = X[:, None, :] - X[None, :, :]
    for i, epsilon in enumerate(eps_values):
        key, subkey = jax.random.split(key)
        G = PointGraph(dx, d, epsilon)
        lap_emp = G.laplacian(f_val, J_true)
        cos_emp = jnp.inner(lap_true, lap_emp) / (jnp.linalg.norm(lap_true)*jnp.linalg.norm(lap_emp))
        error_matrix = error_matrix.at[i, j].set(cos_emp)
        print(f'{i} {epsilon} {j} {n}\r', end='')
print()



print('plotting')

fig, axs = plt.subplots(2, 1,
                        height_ratios=[0.45, 0.7],
                        constrained_layout=True,
                        sharex=True)

N_mesh, Eps_mesh = jnp.meshgrid(
    jnp.append(n_values, n_values[-1]+1e-9), # Pad for the last bin edge
    jnp.append(eps_values, eps_values[-1]+1e-9)
)

mynorm=mcolors.PowerNorm(2, vmin=jnp.nanmin(error_matrix),
                         vmax=jnp.nanmax(error_matrix))

pcm = axs[1].pcolormesh(
    N_mesh, Eps_mesh, error_matrix, 
    shading='flat', 
    cmap='inferno',
    norm=mynorm
)

cbar = fig.colorbar(pcm, ax=axs[1], norm=mynorm)
cbar.set_ticks([0.1, 0.3, 0.5, 0.7, 0.9])
cbar.set_label(r'$\cos$ sim.', fontsize=16)

optimal_eps = eps_values[jnp.nanargmax(error_matrix, axis=0)]
axs[1].plot(n_values, optimal_eps, color='black', linewidth=2.5, 
            label=r'Best $\epsilon(n)$')

axs[1].set_xticks(n_values[::5], labels=[str(n) for n in n_values[::5]])
axs[1].set_xlabel(r'Sampled Points ($n$)', fontsize=18)
axs[1].set_ylabel(r'Bandwidth ($\epsilon$)', fontsize=18)
axs[1].legend(loc='upper right', fontsize=16)

maximum_sim = jnp.nanmax(error_matrix, axis=0)
axs[0].plot(n_values, maximum_sim, color='black', linewidth=2.5, 
            label=r'Best $\epsilon(n)$')
axs[0].set_ylabel(r'$\cos$ sim.', fontsize=18)

plt.savefig('figs/conv.png')
plt.close('all')



us = jnp.linspace(-jnp.pi, jnp.pi, 256, endpoint=True)
vs = jnp.linspace(-jnp.pi, jnp.pi, 128, endpoint=True)
U, V = jnp.meshgrid(us, vs, indexing='ij')
x_UV = (R + r * jnp.cos(V)) * jnp.cos(U)
y_UV = (R + r * jnp.cos(V)) * jnp.sin(U)
z_UV = r * jnp.sin(V)
X_UV = jnp.stack([x_UV, y_UV, z_UV], axis=-1)
f_UV = jax.vmap(jax.vmap(f_ambient))(X_UV)
Lf_UV = jax.vmap(jax.vmap(exact_manifold_laplacian))(X_UV)

f_UV -= jnp.mean(f_UV)
f_UV /= jnp.max(jnp.abs(f_UV))
Lf_UV /= jnp.max(jnp.abs(Lf_UV))

def add_identification_arrows(ax, edge, num_arrows=1, color='black', direction='forward'):
    arrow_length = 0.15
    spacing = 0.05
    
    total_width = (num_arrows - 1) * spacing
    starts = jnp.linspace(-total_width / 2, total_width / 2, num_arrows)
    
    for offset in starts:
        if edge in ['top', 'bottom']:
            x_center = 0.5 + offset
            dx = arrow_length / 2 if direction == 'forward' else -arrow_length / 2
            posA = (x_center - dx, 1.0 if edge == 'top' else 0.0)
            posB = (x_center + dx, 1.0 if edge == 'top' else 0.0)
        else: # left or right
            y_center = 0.5 + offset
            dy = arrow_length / 2 if direction == 'forward' else -arrow_length / 2
            posA = (1.0 if edge == 'right' else 0.0, y_center - dy)
            posB = (1.0 if edge == 'right' else 0.0, y_center + dy)
            
        arrow = patches.FancyArrowPatch(
            posA=posA, posB=posB,
            transform=ax.transAxes,
            color=color,
            arrowstyle='-|>',
            mutation_scale=20,
            linewidth=1.5,
            clip_on=False,
            zorder=10
        )
        ax.add_patch(arrow)

    
plt.imshow(f_UV.T, cmap='coolwarm', vmin=-1, vmax=1)
add_identification_arrows(plt.gca(), 'bottom', num_arrows=2, direction='forward')
add_identification_arrows(plt.gca(), 'top',    num_arrows=2, direction='forward')
add_identification_arrows(plt.gca(), 'left',   num_arrows=1, direction='forward')
add_identification_arrows(plt.gca(), 'right',  num_arrows=1, direction='forward')
plt.gca().axis('off')
plt.savefig('figs/conv_f.png', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close('all')

plt.imshow(Lf_UV.T, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(shrink=0.45)
add_identification_arrows(plt.gca(), 'bottom', num_arrows=2, direction='forward')
add_identification_arrows(plt.gca(), 'top',    num_arrows=2, direction='forward')
add_identification_arrows(plt.gca(), 'left',   num_arrows=1, direction='forward')
add_identification_arrows(plt.gca(), 'right',  num_arrows=1, direction='forward')
plt.gca().axis('off')
plt.savefig('figs/conv_Lf.png', bbox_inches='tight', pad_inches=0.03, dpi=300)
plt.close('all')
