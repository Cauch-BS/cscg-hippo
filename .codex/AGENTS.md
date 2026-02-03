# CSCG-Hippo Project - Codex Agent Documentation

This document provides context for AI coding assistants working on the CSCG-Hippo visualization project. The project aims to create interactive visualizations linking biological hippocampus data with Clone-Structured Cognitive Graph (CSCG) models.

## Project Overview

This project focuses on creating **interactive visualizations** that connect:

- **Biological hippocampus data** (3D UMAP embeddings from zarr cache)
- **CSCG model outputs** (causal graphs, state diagrams, learning dynamics)
- **Dynamic linking** between these visualizations for exploration and analysis

## Visualization Goals

### 1. 3D UMAP Visualization

**Goal**: Create an interactive 3D UMAP visualization based on zarr cache data using Plotly.

**Key Components**:

- Load neural activity data from zarr cache (`vr2p.ExperimentData`)
- Compute 3D UMAP embeddings from spike data
- Create interactive Plotly 3D scatter plots with:
  - Color coding by position markers (Track, Indicator, R1, R2, Teleportation)
  - Trial-based filtering and highlighting
  - Camera position controls
  - Export capabilities (PDF/PNG)
- Reference: `cscg/notebooks/Fig_2g.ipynb`

**Data Sources**:

- Zarr cache files containing multi-session neural data
- Spike data (`spks_big_array`) and fluorescence data (`F_big_array`)
- Position markers, trial information, reward IDs

### 2. Causal Graph Visualization

**Goal**: Display interactive causal graphs learned by CSCG models from maze navigation tasks.

**Key Components**:

- Visualize learned transition graphs using igraph
- Interactive graph layout with Kamada-Kawai algorithm
- Color coding by observation/clone types
- Support for multiple episode visualization
- Reference: `cscg/notebooks/intro.ipynb`

**Data Sources**:

- Trained CSCG models (`CHMM` instances)
- Observation sequences (`x`) and action sequences (`a`)
- Learned transition matrices (`chmm.C` or `chmm.T`)

### 3. CSCG Learning Visualization

**Goal**: Visualize CSCG learning dynamics from the 2ACDC (Two-Alternative Choice Delayed Choice) task.

**Key Components**:

- Track learning progression over iterations
- Visualize transition graph evolution
- Show correlation matrices between trials
- Display state diagram changes during learning
- Reference: `cscg/notebooks/ext_data_fig_8_a_Transition_graph.ipynb`

**Data Sources**:

- Training sequences from 2ACDC task
- CSCG model snapshots at different training iterations
- Forward messages (`mess_fwd`) for state probability analysis

### 4. Dynamic Linking System

**Goal**: Create a unified interactive system that dynamically links:

- 3D UMAP embeddings (biological data)
- 2D state diagrams (CSCG model states)
- Causal graphs (learned transition structures)

**Key Components**:

- Synchronized selection across visualizations
- Cross-highlighting between views
- State-to-embedding mapping
- Interactive exploration of relationships
- Real-time updates when selections change

**Integration Points**:

- Map UMAP points to CSCG states/clones
- Link graph nodes to state diagram regions
- Connect biological neural states to model states

## Technical Stack

### Core Libraries

- **Plotly**: Interactive 3D/2D visualizations
- **ipywidgets**: Interactive controls and widgets
- **igraph**: Graph visualization and layout
- **numpy**: Numerical computations
- **zarr**: Efficient array storage for large datasets
- **vr2p**: Virtual reality 2-photon data processing
- **cscg**: Clone-Structured Cognitive Graph implementation

### Data Formats

- **Zarr**: Hierarchical array storage for neural data
- **NumPy arrays**: Spike data, embeddings, transition matrices
- **Pandas DataFrames**: Position, trial, and metadata

## Project Structure

```text
cscg-hippo/
├── .codex/
│   ├── AGENTS.md          # This file
│   └── SKILLS/            # Codex skills for visualization tasks
│       ├── umap-3d-visualization/
│       ├── causal-graph-visualization/
│       ├── cscg-learning/
│       └── dynamic-linking/
├── cscg/                  # CSCG package
│   ├── notebooks/         # Analysis notebooks
│   │   ├── Fig_2g.ipynb   # 3D UMAP example
│   │   ├── intro.ipynb    # Causal graph examples
│   │   └── ext_data_fig_8_a_Transition_graph.ipynb  # Learning visualization
│   └── cscg/              # Package source
└── vr2p/                  # VR2P data processing package
```

## Key Workflows

### Loading Biological Data

```python
import vr2p
path = '/path/to/data.zarr'
data = vr2p.ExperimentData(path)
# Access multi-session data: data.signals.multi_session.Fns, data.vr, etc.
```

### Training CSCG Models

```python
from cscg import CHMM
chmm = CHMM(n_clones=n_clones, pseudocount=1e-10, x=x, a=a)
progression = chmm.learn_em_T(x, a, n_iter=100)
```

### Creating Interactive Visualizations

- Use Plotly `FigureWidget` for interactive 3D plots
- Combine with ipywidgets for controls
- Use igraph for graph layouts
- Implement callbacks for dynamic linking

## Development Notes

- All visualizations should be interactive and explorable
- Support export to publication-quality formats (PDF, PNG)
- Maintain consistency in color schemes across visualizations
- Ensure performance with large datasets (use efficient data structures)
- Follow existing notebook patterns for consistency

## References

- CSCG Paper: ["Learning cognitive maps as structured graphs for vicarious evaluation"](https://www.biorxiv.org/content/10.1101/864421v4.full)
- Existing notebooks provide templates and examples for each visualization type
- See individual skill documentation in `SKILLS/` for detailed implementation guidance
