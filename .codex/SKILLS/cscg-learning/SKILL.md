---
name: cscg-learning
description: Visualize CSCG learning dynamics and progression from training tasks like 2ACDC. Use when tracking learning progression, visualizing transition graph evolution, analyzing trial correlations, displaying state diagram changes during training, or working with forward messages and state probabilities.
---

# CSCG Learning Visualization

Visualize how Clone-Structured Cognitive Graphs learn and evolve during training, particularly for tasks like 2ACDC (Two-Alternative Choice Delayed Choice).

## Quick Start

Track learning progression:

```python
from cscg import CHMM
import numpy as np
import matplotlib.pyplot as plt

# Initialize model
chmm = CHMM(n_clones=n_clones, pseudocount=1e-10, x=x, a=a)

# Train and track progression
progression = chmm.learn_em_T(x, a, n_iter=100, term_early=False)

# Plot learning curve
plt.plot(progression)
plt.xlabel('Iteration')
plt.ylabel('Negative Log-Likelihood')
```

## Learning Progression Tracking

### Basic Training Loop

```python
def train_with_tracking(chmm, x, a, n_iter=100, snapshot_iters=[0, 10, 25, 50, 100]):
    """Train CHMM and capture snapshots at specified iterations."""
    snapshots = []

    for i in range(n_iter):
        progression = chmm.learn_em_T(x, a, n_iter=1, term_early=False)

        if i in snapshot_iters:
            # Save model snapshot
            snapshots.append({
                'iteration': i,
                'chmm': copy.deepcopy(chmm),
                'nll': progression[-1] if len(progression) > 0 else None
            })

    return snapshots
```

### Refinement Phase

After EM training, refine with Viterbi:

```python
# Initial EM training
chmm.pseudocount = 2e-3
progression = chmm.learn_em_T(x, a, n_iter=1000)

# Refinement with Viterbi
chmm.pseudocount = 0.0
chmm.learn_viterbi_T(x, a, n_iter=100)
```

## Forward Messages and State Probabilities

### Computing Forward Messages

```python
def get_mess_fwd(chmm, x, pseudocount=0.0, pseudocount_E=0.1):
    """Compute forward messages for state probability analysis."""
    from cscg import forwardE

    n_clones = chmm.n_clones
    E = np.zeros((n_clones.sum(), len(n_clones)))
    last = 0
    for c in range(len(n_clones)):
        E[last : last + n_clones[c], c] = 1
        last += n_clones[c]

    E += pseudocount_E
    norm = E.sum(1, keepdims=True)
    norm[norm == 0] = 1
    E /= norm

    T = chmm.C + pseudocount
    norm = T.sum(2, keepdims=True)
    norm[norm == 0] = 1
    T /= norm
    T = T.mean(0, keepdims=True)

    log2_lik, mess_fwd = forwardE(
        T.transpose(0, 2, 1),
        E,
        chmm.Pi_x,
        chmm.n_clones,
        x,
        x * 0,
        store_messages=True
    )

    return mess_fwd
```

### State Probability Analysis

```python
# Compute forward messages
mess_fwd = get_mess_fwd(chmm, x, pseudocount_E=0.1)

# Shape: (n_timepoints, n_clones_total)
# Each row is probability distribution over clones at that timepoint

# Find most active clones
clone_activity = mess_fwd.sum(0)  # Total activity per clone
top_clones = np.argsort(clone_activity)[-10:]  # Top 10 most active
```

## Trial Correlation Analysis

### Computing Trial Correlations

```python
import scipy.stats

def compute_trial_correlations(mess_fwd, trials, tr_len):
    """Compute correlation matrix between trials."""
    n_trials = len(trials)
    corrplot = np.zeros((n_trials, n_trials))

    for trial1 in range(n_trials):
        for trial2 in range(trial1, n_trials):
            comp1 = mess_fwd[trial1 * tr_len : (trial1 + 1) * tr_len, :]
            comp2 = mess_fwd[trial2 * tr_len : (trial2 + 1) * tr_len, :]
            corrplot[trial1, trial2] = scipy.stats.pearsonr(
                comp1.flatten(),
                comp2.flatten()
            )[0]

    # Fill symmetric part
    corrplot = corrplot + corrplot.T - np.diag(np.diag(corrplot))
    return corrplot
```

### Visualizing Correlations

```python
import seaborn as sb

corrplot = compute_trial_correlations(mess_fwd, trials, tr_len)
sb.heatmap(corrplot, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Trial-to-Trial Correlation Matrix')
plt.xlabel('Trial')
plt.ylabel('Trial')
```

### Cross-Trial Correlations

```python
# Compare specific trials
trial1 = mess_fwd[-tr_len:, :]  # Last trial
trial0 = mess_fwd[-2*tr_len:-tr_len, :]  # Second-to-last trial

# Position-to-position correlation
corr_plot01 = np.zeros((tr_len, tr_len))
for posi in range(tr_len):
    for posj in range(tr_len):
        corr_plot01[posi, posj] = scipy.stats.pearsonr(
            trial0[posi], trial1[posj]
        )[0]

sb.heatmap(corr_plot01, cmap='coolwarm')
plt.title('Cross-Trial Position Correlation')
```

## Transition Graph Evolution

### Visualizing Graph Changes

```python
import igraph
import copy

def visualize_learning_progression(chmm_snapshots, x, a, output_dir='figures'):
    """Create transition graphs at different training stages."""
    for snapshot in chmm_snapshots:
        chmm_curr = snapshot['chmm']
        iter_num = snapshot['iteration']

        # Extract and plot graph
        states = chmm_curr.decode(x, a)[1]
        v = np.unique(states)
        T = chmm_curr.C[:, v][:, :, v]
        A = T.sum(0)
        A /= A.sum(1, keepdims=True)

        g = igraph.Graph.Adjacency((A > 0).tolist())

        igraph.plot(
            g,
            f"{output_dir}/graph_iter_{iter_num}.pdf",
            layout=g.layout("kamada_kawai"),
            vertex_size=20
        )
```

### Tracking Graph Statistics

```python
def track_graph_stats(chmm_snapshots, x, a):
    """Track graph statistics over training."""
    stats = {
        'iterations': [],
        'n_states': [],
        'n_edges': [],
        'avg_degree': []
    }

    for snapshot in chmm_snapshots:
        chmm_curr = snapshot['chmm']
        states = chmm_curr.decode(x, a)[1]
        v = np.unique(states)

        T = chmm_curr.C[:, v][:, :, v]
        A = T.sum(0)
        A /= A.sum(1, keepdims=True)

        g = igraph.Graph.Adjacency((A > 0).tolist())

        stats['iterations'].append(snapshot['iteration'])
        stats['n_states'].append(len(v))
        stats['n_edges'].append(g.ecount())
        stats['avg_degree'].append(np.mean(g.degree()))

    return stats
```

## 2ACDC Task Specific

### Trial Structure

```python
# Define trial structure for 2ACDC task
trial1x_let = np.repeat(np.array([
    'A','A','A','A','A','A','B', 'B', 'B', 'B', 'A','A','A',
    'D','F','A','A','A','E','E', 'A', 'A', 'G','H','H','H'
]), 1)

trial2x_let = np.repeat(np.array([
    'A','A','A','A','A','A','C', 'C', 'C', 'C', 'A','A','A',
    'D','D','A','A','A','E','F', 'A', 'A','G','H','H','H'
]), 1)

# Map letters to numbers
letter_num_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
```

### Training on 2ACDC Data

```python
# Create sequence from multiple trials
num_trials = 5
trials = np.random.choice(2, num_trials - 2)
trials = np.concatenate((trials, np.array([0, 1])))

tr_len = len(trial1x)
x = np.zeros(num_trials * tr_len, dtype=np.int64)

for trial in range(len(trials)):
    if trials[trial] == 0:
        x[trial * tr_len : (trial + 1) * tr_len] = trial1x
    else:
        x[trial * tr_len : (trial + 1) * tr_len] = trial2x

a = np.zeros(len(x), dtype=np.int64)
OBS = len(np.unique(x))

# Initialize and train
n_clones = np.ones(OBS + 5, dtype=np.int64) * 100
chmm = CHMM(n_clones=n_clones, pseudocount=1e-10, x=x, a=a, seed=0)

# Track learning with snapshots
snapshots = []
for tot_iter in range(0, 26):
    n_iter = 10
    progression = chmm.learn_em_T(x, a, n_iter=n_iter, term_early=False)

    if tot_iter in [0, 5, 10, 25]:
        mess_fwd = get_mess_fwd(chmm, x, pseudocount_E=0.1)
        # Compute correlations, visualize, etc.
        snapshots.append({
            'iteration': tot_iter * n_iter,
            'chmm': copy.deepcopy(chmm),
            'mess_fwd': mess_fwd
        })
```

## Visualization Dashboard

### Combined Learning View

```python
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def create_learning_dashboard(snapshots, mess_fwd_history, corr_history):
    """Create multi-panel dashboard showing learning progression."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig)

    # Learning curve
    ax1 = fig.add_subplot(gs[0, :])
    iterations = [s['iteration'] for s in snapshots]
    nlls = [s['nll'] for s in snapshots]
    ax1.plot(iterations, nlls)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Negative Log-Likelihood')
    ax1.set_title('Learning Curve')

    # Correlation evolution
    ax2 = fig.add_subplot(gs[1, 0])
    # Plot correlation matrices at different iterations

    # Graph evolution
    ax3 = fig.add_subplot(gs[1, 1])
    # Show graph statistics over time

    # State probability heatmap
    ax4 = fig.add_subplot(gs[2, :])
    # Show mess_fwd as heatmap

    plt.tight_layout()
    return fig
```

## Reference Implementation

See `cscg/notebooks/ext_data_fig_8_a_Transition_graph.ipynb` for complete example including:

- 2ACDC trial structure
- Learning progression tracking
- Trial correlation analysis
- Transition graph visualization at multiple stages
- Cross-trial position correlations
