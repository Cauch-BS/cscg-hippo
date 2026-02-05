---
name: causal-graph-visualization
description: Visualize causal graphs learned by CSCG models from maze navigation tasks. Use when working with CHMM models, transition graphs, igraph visualization, cognitive maps, or displaying learned state transition structures from CSCG training.
---

# Causal Graph Visualization for CSCG Models

Visualize the learned transition graphs (causal structures) from trained Clone-Structured Cognitive Graph (CSCG) models.

## Quick Start

Visualize a learned transition graph:

```python
from cscg import CHMM
import igraph
import numpy as np

# After training a CHMM
chmm = CHMM(n_clones=n_clones, pseudocount=1e-10, x=x, a=a)
chmm.learn_em_T(x, a, n_iter=100)

# Decode states and visualize graph
states = chmm.decode(x, a)[1]
v = np.unique(states)
T = chmm.C[:, v][:, :, v]
A = T.sum(0)
A /= A.sum(1, keepdims=True)

g = igraph.Graph.Adjacency((A > 0).tolist())
igraph.plot(g, output_file="graph.pdf", layout=g.layout("kamada_kawai"))
```

## Graph Extraction

### From Trained CHMM

```python
def extract_transition_graph(chmm, x, a, multiple_episodes=False):
    """Extract transition graph from trained CHMM."""
    states = chmm.decode(x, a)[1]
    v = np.unique(states)

    if multiple_episodes:
        # For multi-episode data, exclude first state
        T = chmm.C[:, v][:, :, v][:-1, 1:, 1:]
        v = v[1:]
    else:
        T = chmm.C[:, v][:, :, v]

    # Aggregate over actions and normalize
    A = T.sum(0)  # Sum over actions
    A /= A.sum(1, keepdims=True)  # Normalize rows

    return A, v
```

### Graph Construction

```python
import igraph

A, v = extract_transition_graph(chmm, x, a)
g = igraph.Graph.Adjacency((A > 0).tolist())  # Binary adjacency
```

## Visualization

### Basic Graph Plot

```python
import matplotlib.pyplot as plt
from matplotlib import cm, colors

def plot_graph(chmm, x, a, output_file, cmap=cm.Spectral,
               multiple_episodes=False, vertex_size=30):
    """Plot learned cognitive map as a graph."""
    states = chmm.decode(x, a)[1]
    v = np.unique(states)

    if multiple_episodes:
        T = chmm.C[:, v][:, :, v][:-1, 1:, 1:]
        v = v[1:]
    else:
        T = chmm.C[:, v][:, :, v]

    A = T.sum(0)
    A /= A.sum(1, keepdims=True)

    g = igraph.Graph.Adjacency((A > 0).tolist())

    # Color nodes by observation/clone
    node_labels = np.arange(x.max() + 1).repeat(n_clones)[v]
    if multiple_episodes:
        node_labels -= 1

    colors = [cmap(nl)[:3] for nl in node_labels / node_labels.max()]

    igraph.plot(
        g,
        output_file,
        layout=g.layout("kamada_kawai"),
        vertex_color=colors,
        vertex_label=v,
        vertex_size=vertex_size,
        margin=50,
    )
```

### Layout Algorithms

**Kamada-Kawai** (default, good for general graphs):

```python
layout = g.layout("kamada_kawai")
```

**Fruchterman-Reingold** (force-directed):

```python
layout = g.layout("fr")
```

**Circular** (for small graphs):

```python
layout = g.layout("circle")
```

**Grid** (for structured layouts):

```python
layout = g.layout("grid")
```

## Color Schemes

### By Observation Type

```python
from matplotlib import cm, colors

# Map states to observations
n_clones = chmm.n_clones
node_labels = np.arange(x.max() + 1).repeat(n_clones)[v]

# Use colormap
cmap = cm.Spectral
colors = [cmap(nl / node_labels.max())[:3] for nl in node_labels]
```

### Custom Color Mapping

```python
custom_colors = np.array([
    [214, 214, 214],  # Gray
    [85, 35, 157],    # Purple
    [253, 252, 144],  # Yellow
    [114, 245, 144], # Green
    # ... more colors
]) / 256

cmap = colors.ListedColormap(custom_colors)
colors = [cmap(nl)[:3] for nl in node_labels / node_labels.max()]
```

### By Clone ID

```python
# Color by clone index within observation
clone_indices = v % n_clones.sum()
colors = [cmap(ci / clone_indices.max())[:3] for ci in clone_indices]
```

## Interactive Visualization

### Using Plotly

```python
import plotly.graph_objects as go
import networkx as nx

# Convert igraph to networkx for Plotly compatibility
G_nx = nx.Graph()
for edge in g.es:
    G_nx.add_edge(edge.source, edge.target)

# Get layout positions
pos = {i: (layout[i][0], layout[i][1]) for i in range(len(v))}

# Create edges
edge_x = []
edge_y = []
for edge in G_nx.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines'
)

# Create nodes
node_x = [pos[i][0] for i in range(len(v))]
node_y = [pos[i][1] for i in range(len(v))]

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    marker=dict(
        size=10,
        color=colors,
        line=dict(width=2)
    ),
    text=[str(vi) for vi in v],
    hoverinfo='text'
)

fig = go.FigureWidget(data=[edge_trace, node_trace])
fig.update_layout(showlegend=False, hovermode='closest')
```

## Multi-Episode Graphs

For data with multiple episodes (e.g., stitched rooms):

```python
# Mark episode boundaries
x = np.hstack((0, x1 + 1, 0, x2 + 1))  # 0 marks episode boundary
a = np.hstack((4, a1[:-1], 4, 4, a2))  # Action 4 = episode transition

# Visualize with episode markers
graph = plot_graph(
    chmm, x, a,
    output_file="multi_episode_graph.pdf",
    multiple_episodes=True
)
```

## Graph Analysis

### Transition Probabilities

```python
# Extract transition probabilities
A, v = extract_transition_graph(chmm, x, a)

# Find strongly connected components
g = igraph.Graph.Adjacency((A > 0).tolist())
components = g.components(mode='strong')
print(f"Number of components: {len(components)}")

# Compute centrality measures
betweenness = g.betweenness()
closeness = g.closeness()
```

### State Statistics

```python
# Count transitions per state
out_degrees = A.sum(1)
in_degrees = A.sum(0)

# Find hub states (high degree)
hub_threshold = np.percentile(out_degrees, 90)
hub_states = v[out_degrees > hub_threshold]
```

## Styling Options

### Vertex Properties

```python
igraph.plot(
    g,
    output_file,
    layout=layout,
    vertex_color=colors,
    vertex_label=v,
    vertex_size=30,           # Size of nodes
    vertex_label_size=12,     # Label font size
    vertex_label_dist=1.5,    # Distance of label from node
    margin=50,                # Plot margin
    bbox=(800, 800)          # Figure size
)
```

### Edge Properties

```python
# Weight edges by transition probability
edge_weights = []
for i in range(len(v)):
    for j in range(len(v)):
        if A[i, j] > 0:
            edge_weights.append(A[i, j])

g.es['weight'] = edge_weights
g.es['width'] = [w * 5 for w in edge_weights]  # Scale for visibility
```

## Reference Implementation

See `cscg/notebooks/intro.ipynb` for examples including:

- Rectangular room navigation
- Empty room navigation
- Stitched multi-room environments
- Place field visualization alongside graphs
