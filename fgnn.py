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

key = jr.PRNGKey(42) # RNG
    
b_true = 0.5 * jnp.array([0.5, 0.5]) # drift vector for Randers metric
T_int = 1.0 # integration time for equation

n_train, n_test, d = 600, 600, 2 # sizes of graphs for training and testing
eps = 0.2 # bandwidth for graph creation

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration for the graph neural diffusion model.")
    
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate.")
    parser.add_argument('--steps', type=int, default=5000, help="Training steps.")
    parser.add_argument('--M', type=int, default=1, help="Number of initial-final function pairs / batch size.")
    parser.add_argument('--mask_ratio', type=float, default=0.20, help="Ratio of nodes to mask for the SSL task.")
    parser.add_argument('--hidden_dim', type=int, default=64, help="Hidden dimension size of the neural architecture.")
    parser.add_argument('--euclidean', action=argparse.BooleanOptionalAction, default=True, 
                        help="Use 2-norm. If not, use 3-norm. Use --no-euclidean to disable.")

    parser.add_argument('--use_relu', action=argparse.BooleanOptionalAction, default=True, 
                        help="Enable ReLU activation. Use --no-use_relu to disable.")
    
    parser.add_argument('--use_baseline', action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Use standard graph neural network.")
    
    # Strings
    parser.add_argument('--expid', type=str, default='000', help="Experiment ID for logging.")

    return parser.parse_args()

args = parse_args()

lr = args.lr
steps = args.steps
M = args.M
mask_ratio = args.mask_ratio
hidden_dim = args.hidden_dim
use_relu = args.use_relu
use_baseline = args.use_baseline
euclidean = args.euclidean
expid = args.expid

with open(f'figs/fgnn_{expid}.params', 'w') as f:
    print(f"Running Experiment {expid} with LR={lr}, Steps={steps}")
    f.write(f'use_baseline={use_baseline}\n')
    f.write(f'lr={lr}\nsteps={steps}\nm={M}\nmask_ratio={mask_ratio}\n')
    f.write(f'hidden_dim={hidden_dim}\nuse_relu={use_relu}\n')
    
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
    p: int

    def __init__(self, b, p=2):
        self.b = b
        self.p = p

    def E(self, xi):
        norm = (jnp.sum(jnp.abs(xi)**self.p))**(1/self.p)
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
        # Tikhonov pseudoinverse
        C /= (self.n * self.eps**(self.d+2))
        self.C_inv = jnp.linalg.inv(C + eps_pinv * jnp.eye(self.D))

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

class GCN(eqx.Module):
    v_lift: jax.Array
    v_proj: jax.Array
    mlp: eqx.nn.MLP

    def __init__(self, d, hidden_dim, activ=jax.nn.relu, key=jr.PRNGKey(0)):
        k0, k1, k2 = jr.split(key, 3)
        self.v_lift = jr.normal(k0, hidden_dim)
        self.v_proj = jr.normal(k1, hidden_dim)
        self.mlp = eqx.nn.MLP(in_size=hidden_dim, out_size=hidden_dim,
                              width_size = hidden_dim, depth=1,
                              activation=activ,
                              key=k2)

    def lift(self, f):
        return f[...,None] * self.v_lift

    def project(self, F):
        return F @ self.v_proj

    def laplacian(self, G, F):
        dF = F[:, None, :] - F[None, :, :]
        K_G = G.K
        lap = jnp.sum(K_G[:,:,None] * dF, axis=1) / G.n
        return jax.vmap(self.mlp)(lap)

# ==========================================
# ODE Integrator
# ==========================================

def simulate_diffrax(G, f0, t_final, laplacian_fn):
    def vector_field(t, f, _):
        return -laplacian_fn(G, f)
    
    term = ODETerm(vector_field)
    solver = Tsit5()
    
    sol = diffeqsolve(
        term, solver,
        t0=0.0, t1=t_final, dt0=0.05,
        y0=f0,
        saveat=SaveAt(t1=True),
        adjoint=RecursiveCheckpointAdjoint(),
    )
    return sol.ys[0]

# ==========================================
# Synthetic Data Generation
# ==========================================

def make_random_fourier_f0(key, X, num_modes=24):
    d = X.shape[-1]
    k1, k2, k3 = jax.random.split(key, 3)
    
    freqs = jax.random.uniform(k1, (num_modes, d), minval=1.0, maxval=10.0)
    phases = jax.random.uniform(k2, (num_modes,), minval=0.0, maxval=2*jnp.pi)
    amps = jax.random.normal(k3, (num_modes,))
    
    def wave_val(x):
        return jnp.sum(amps * jnp.sin(jnp.dot(freqs, x) * 2 * jnp.pi + phases))
        
    f0 = jax.vmap(wave_val)(X)
    return (f0 - jnp.min(f0)) / (jnp.max(f0) - jnp.min(f0))

# ==========================================
# SETUP BEGINS HERE
# ==========================================

print('Building graphs')

key, k0, k1 = jr.split(key, 3)
X_train = jax.random.uniform(k0, (n_train, d))
dx_train = X_train[:, None, :] - X_train[None, :, :]
dx_train = dx_train - jnp.round(dx_train)

X_test = jax.random.uniform(k1, (n_test, d))
dx_test = X_test[:, None, :] - X_test[None, :, :]
dx_test = dx_test - jnp.round(dx_test)
del k0, k1

G_train = PointGraph(dx_train, d, eps)
G_test = PointGraph(dx_test, d, eps)
    
print('Generating initial conditions')

key, k0, k1 = jr.split(key, 3)
f0_train_keys = jr.split(k0, M)
f0_train = jax.vmap(make_random_fourier_f0, in_axes=(0, None))(f0_train_keys, X_train)
f0_test_keys = jr.split(k1, (16, M))
f0_test = jax.vmap(jax.vmap(make_random_fourier_f0,
                            in_axes=(0, None)),
                   in_axes=(0,None))(f0_test_keys, X_test)
del k0, k1, f0_train_keys, f0_test_keys

print('Generating target data')

if euclidean:
    true_metric = RandersMetric(b_true)
else:
    true_metric = RandersMetric(b_true, p=3)

true_lap_fn = lambda G, f: G.laplacian(f, true_metric.J)

# def                simulate_diffrax           G,    f, t_fi, lap):
sim_batch = jax.vmap(simulate_diffrax, in_axes=(None, 0, None, None))
fT_train = sim_batch(G_train, f0_train, T_int, true_lap_fn)
fT_test = jax.vmap(sim_batch, in_axes=(None, 0, None, None))(G_test, f0_test, T_int, true_lap_fn)

print('Initializing neural model and defining loss/eval functions')

num_labeled = int(mask_ratio * n_train)
mask_U = jnp.zeros(n_train).at[:num_labeled].set(1.0)

key, k0 = jr.split(key)

activ = jax.nn.identity
if use_relu:
    activ = jax.nn.relu

post_activ = jax.nn.identity
if use_baseline:
    post_activ = jax.nn.relu


if use_baseline:

    neural_metric = GCN(d, hidden_dim, activ, key=k0)
    
    def loss(model):
        F0 = model.lift(f0_train)
        FT = sim_batch(G_train, F0, T_int,
                       lambda G, F: -model.laplacian(G, F))
        f_pred = model.project(FT)
        mse = jnp.sum(mask_U * (f_pred - fT_train)**2, axis=-1) / jnp.sum(mask_U * fT_train**2, axis=-1)
        return jnp.mean(mse)
    
else:
    
    neural_metric = SymmetricNeuralMetric(d, hidden_dim, activ, key=k0)
    model_lap_fn = lambda model, G, f: G.laplacian(f, model.J)

    def loss(model):
        lap_fn = lambda G, f: G.laplacian(f, model.J)
        f_pred = sim_batch(G_train, f0_train, T_int, lap_fn)
        mse = jnp.sum(mask_U * (f_pred - fT_train)**2, axis=-1) / jnp.sum(mask_U * fT_train**2, axis=-1)
        return jnp.mean(mse)


    
del k0



if use_baseline:
    def eval_model(model, G, f0_batch, fT_batch, mask=1.):
        F0 = model.lift(f0_batch)
        FT = sim_batch(G, F0, T_int,
                       lambda G, F: -model.laplacian(G, F))
        f_pred = model.project(FT)
        return jnp.mean(mask*(f_pred - fT_batch)**2) / jnp.mean(mask*fT_batch**2)
else:
    def eval_model(model, G, f0_batch, fT_batch, mask=1.):
        lap_fn = lambda G, f: model_lap_fn(model, G, f)
        f_pred = sim_batch(G, f0_batch, T_int, lap_fn)
        return jnp.mean(mask*(f_pred - fT_batch)**2) / jnp.mean(mask*fT_batch**2)

print('Initializing optimizer and defining training loop')

optim = optax.adamw(lr)
opt_state = optim.init(eqx.filter(neural_metric, eqx.is_array))

@eqx.filter_jit
def make_step(
        model,
        opt_state,
):
    loss_value, grads = eqx.filter_value_and_grad(loss)(model)
    updates, opt_state = optim.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value

print('Training model')

for step in range(1, steps+1):
    neural_metric, opt_state, loss_value = make_step(neural_metric, opt_state)
    if step % 100 == 0:
        print(f"Step {step:3d} | Loss {loss_value:.8f}\r", end='')
print()

# ==========================================
# Evaluation
# ==========================================

print('Evaluating')

eval_train = eval_model(neural_metric, G_train, f0_train, fT_train, 1-mask_U)
eval_test = jax.vmap(eval_model, in_axes=(None, None, 0, 0))(neural_metric, G_test, f0_test, fT_test)
eval_test_std = jnp.std(eval_test)
eval_test = jnp.mean(eval_test)

with open(f'figs/fgnn_{expid}_results.txt', 'w') as f:
    f.write('=====================\n')
    f.write('       Results\n')
    f.write('---------------------\n')
    f.write(f'Train graph       : {eval_train:.2g}\n')
    f.write(f'Test graph (mean) : {eval_test:.2g}\n')
    f.write(f'Test graph (std.) : {eval_test_std:.2g}\n')
    f.write('=====================\n')

# ==========================================
# Visualization
# ==========================================

print('Visualizing')

if use_baseline:
    exit()

thetas = jnp.linspace(-jnp.pi, jnp.pi, 32, endpoint=True)
xi_thetas = jnp.array((jnp.cos(thetas), jnp.sin(thetas))).T

E_thetas_true = jax.vmap(true_metric.E)(xi_thetas)
E_thetas_model = jax.vmap(neural_metric.E)(xi_thetas)

E_thetas_model /= jnp.min(E_thetas_true)
E_thetas_true /= jnp.min(E_thetas_true)

with open(f'figs/fgnn_{expid}_ball.csv', 'w') as f:
    f.write('theta,rtrue,rmodel\n')
    for theta, E_t, E_m in zip(thetas, E_thetas_true, E_thetas_model):
        f.write(f'{theta},{1./E_t},{1./E_m}\n')

grid_val = jnp.linspace(-1, 1, 100)
Xi_grid = jnp.array(jnp.meshgrid(grid_val, grid_val)).transpose(1, 2, 0)

E_true = jax.vmap(jax.vmap(true_metric.E))(Xi_grid)
E_model = jax.vmap(jax.vmap(neural_metric.E))(Xi_grid)

# Normalize energies due to time/scale degeneracy to compare shapes
E_true_norm = E_true / jnp.max(E_true)
E_model_norm = E_model / jnp.max(E_model)
    
plt.figure(figsize=(6, 6))
plt.title("Recovered Finsler Metric", fontsize=14)
    
plt.contour(Xi_grid[...,0], Xi_grid[...,1], E_true_norm, levels=[0.1],
            colors='black', linewidths=2, linestyles='dashed')
plt.plot([], [], color='black', linestyle='dashed', linewidth=2,
         label='true')
plt.contour(Xi_grid[...,0], Xi_grid[...,1], E_model_norm, levels=[0.1],
            colors='blue', linewidths=2)
plt.plot([], [], color='blue', linewidth=2,
         label='learned')
    
plt.scatter([0], [0], color='red', marker='x')
plt.axhline(0, color='gray', alpha=0.3)
plt.axvline(0, color='gray', alpha=0.3)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.2)
plt.axis('equal')
plt.savefig(f'figs/fgnn_ball_{expid}.png')
plt.close('all')
