---
name: dynamic-linking
description: Create dynamically linked interactive visualizations connecting 3D UMAP embeddings, 2D state diagrams, and causal graphs from CSCG models. Use when building unified visualization systems, synchronizing selections across multiple plots, mapping biological states to model states, or creating interactive exploration tools that link different data views.
---

# Dynamic Linking System

Create a unified interactive system that dynamically links 3D UMAP embeddings (biological data), 2D state diagrams (CSCG model states), and causal graphs (learned transition structures).

## Quick Start

Set up basic linking between visualizations:

```python
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display

# Create multiple figure widgets
fig_umap = go.FigureWidget(...)  # 3D UMAP plot
fig_state = go.FigureWidget(...)  # 2D state diagram
fig_graph = go.FigureWidget(...)  # Causal graph

# Link selection events
def on_umap_selection(trace, points, state):
    selected_indices = points.point_inds
    # Update other visualizations
    highlight_in_state_diagram(selected_indices)
    highlight_in_graph(selected_indices)

fig_umap.data[0].on_selection(on_umap_selection)
```

## Core Linking Architecture

### Shared State Management

```python
class VisualizationLinker:
    """Manages synchronized state across multiple visualizations."""
    
    def __init__(self):
        self.selected_indices = []
        self.umap_embedding = None
        self.state_mapping = None  # Maps UMAP points to CSCG states
        self.graph_nodes = None
        
    def set_umap_data(self, embedding, metadata):
        """Set UMAP embedding data."""
        self.umap_embedding = embedding
        self.umap_metadata = metadata
        
    def set_state_mapping(self, mapping):
        """Set mapping from UMAP indices to CSCG state/clone IDs."""
        # mapping: dict {umap_idx: state_id} or array
        self.state_mapping = mapping
        
    def set_graph_data(self, graph, node_to_state):
        """Set graph data and node-to-state mapping."""
        self.graph = graph
        self.node_to_state = node_to_state
        
    def select_by_umap_indices(self, indices):
        """Select points in UMAP and propagate to other views."""
        self.selected_indices = indices
        self.update_all_views()
        
    def select_by_state_ids(self, state_ids):
        """Select by CSCG state/clone IDs."""
        # Find UMAP indices corresponding to states
        umap_indices = self.find_umap_for_states(state_ids)
        self.select_by_umap_indices(umap_indices)
        
    def update_all_views(self):
        """Update all visualizations based on current selection."""
        self.update_umap_highlight()
        self.update_state_diagram_highlight()
        self.update_graph_highlight()
```

## UMAP to State Mapping

### Creating State Mapping

```python
def create_state_mapping(umap_embedding, chmm, x, a, selected_position):
    """Map UMAP points to CSCG states/clones."""
    # Decode states for the sequence
    states = chmm.decode(x, a)[1]
    
    # Create mapping: for each timepoint, map to state
    # Assuming umap_embedding corresponds to same timepoints as x
    mapping = {}
    for i in range(len(umap_embedding)):
        state_id = states[i]
        clone_id = state_id  # Or extract clone ID from state
        mapping[i] = {
            'state_id': state_id,
            'clone_id': clone_id,
            'observation': x[i],
            'position_marker': selected_position.iloc[i]['position_marker']
        }
    
    return mapping
```

### Reverse Mapping (State to UMAP)

```python
def find_umap_indices_for_state(mapping, state_id):
    """Find all UMAP indices corresponding to a given state."""
    indices = []
    for idx, data in mapping.items():
        if data['state_id'] == state_id:
            indices.append(idx)
    return indices

def find_umap_indices_for_clone(mapping, clone_id):
    """Find all UMAP indices corresponding to a given clone."""
    indices = []
    for idx, data in mapping.items():
        if data['clone_id'] == clone_id:
            indices.append(idx)
    return indices
```

## Selection Propagation

### UMAP Selection → State Diagram

```python
def on_umap_selection(trace, points, state):
    """Handle selection in UMAP plot."""
    linker = state['linker']
    selected_indices = points.point_inds
    
    # Get corresponding states
    selected_states = set()
    for idx in selected_indices:
        if idx in linker.state_mapping:
            selected_states.add(linker.state_mapping[idx]['state_id'])
    
    # Highlight in state diagram
    highlight_states_in_diagram(linker.fig_state, selected_states)
    
    # Highlight in graph
    highlight_nodes_in_graph(linker.fig_graph, selected_states)
```

### Graph Selection → UMAP

```python
def on_graph_selection(trace, points, state):
    """Handle selection in graph plot."""
    linker = state['linker']
    selected_nodes = points.point_inds
    
    # Get states corresponding to selected nodes
    selected_states = [linker.node_to_state[node_idx] 
                       for node_idx in selected_nodes]
    
    # Find UMAP indices
    umap_indices = []
    for state_id in selected_states:
        umap_indices.extend(
            find_umap_indices_for_state(linker.state_mapping, state_id)
        )
    
    # Highlight in UMAP
    highlight_points_in_umap(linker.fig_umap, umap_indices)
    
    # Highlight in state diagram
    highlight_states_in_diagram(linker.fig_state, selected_states)
```

## Highlighting Functions

### Highlight UMAP Points

```python
def highlight_points_in_umap(fig, indices, highlight_color='red', 
                              highlight_size=5):
    """Highlight selected points in UMAP plot."""
    # Create highlight trace
    if len(fig.data) > 1:
        # Update existing highlight trace
        fig.data[1].x = [fig.data[0].x[i] for i in indices]
        fig.data[1].y = [fig.data[0].y[i] for i in indices]
        fig.data[1].z = [fig.data[0].z[i] for i in indices]
    else:
        # Add new highlight trace
        highlight_trace = go.Scatter3d(
            x=[fig.data[0].x[i] for i in indices],
            y=[fig.data[0].y[i] for i in indices],
            z=[fig.data[0].z[i] for i in indices],
            mode='markers',
            marker=dict(
                size=highlight_size,
                color=highlight_color,
                opacity=0.8,
                line=dict(width=2, color='black')
            ),
            name='Selected',
            showlegend=False
        )
        fig.add_trace(highlight_trace)
```

### Highlight Graph Nodes

```python
def highlight_nodes_in_graph(fig, state_ids, highlight_color='red'):
    """Highlight nodes in graph corresponding to selected states."""
    # Update node colors based on selection
    node_colors = []
    for i, state_id in enumerate(fig.data[1].customdata):
        if state_id in state_ids:
            node_colors.append(highlight_color)
        else:
            node_colors.append(fig.data[1].marker.color[i])
    
    fig.data[1].marker.color = node_colors
```

### Highlight State Diagram Regions

```python
def highlight_states_in_diagram(fig, state_ids, highlight_alpha=0.5):
    """Highlight regions in 2D state diagram."""
    # Update opacity or color of selected state regions
    for i, trace in enumerate(fig.data):
        if hasattr(trace, 'customdata') and trace.customdata:
            state_id = trace.customdata.get('state_id')
            if state_id in state_ids:
                trace.marker.opacity = highlight_alpha
                trace.marker.line.width = 3
```

## Interactive Controls

### Unified Control Panel

```python
def create_linking_controls(linker):
    """Create control panel for linked visualizations."""
    
    # Selection mode
    selection_mode = widgets.RadioButtons(
        options=['UMAP', 'Graph', 'State Diagram', 'All'],
        value='All',
        description='Select from:'
    )
    
    # Filter by state
    state_selector = widgets.IntRangeSlider(
        value=[0, linker.n_states - 1],
        min=0,
        max=linker.n_states - 1,
        step=1,
        description='State range:'
    )
    
    # Filter by observation
    obs_selector = widgets.Dropdown(
        options=list(range(linker.n_observations)),
        description='Observation:'
    )
    
    def update_selection(change):
        if selection_mode.value == 'All':
            # Select all points
            linker.select_by_umap_indices(list(range(len(linker.umap_embedding))))
        elif selection_mode.value == 'Graph':
            # Enable graph selection
            pass
        # ... handle other modes
    
    selection_mode.observe(update_selection, names='value')
    
    return widgets.VBox([
        selection_mode,
        state_selector,
        obs_selector
    ])
```

## Complete Linked System

### Setup Function

```python
def create_linked_visualization_system(umap_embedding, chmm, x, a, 
                                       selected_position, graph_data):
    """Create complete linked visualization system."""
    
    # Initialize linker
    linker = VisualizationLinker()
    linker.set_umap_data(umap_embedding, selected_position)
    
    # Create state mapping
    state_mapping = create_state_mapping(
        umap_embedding, chmm, x, a, selected_position
    )
    linker.set_state_mapping(state_mapping)
    
    # Create visualizations
    fig_umap = create_umap_plot(umap_embedding, selected_position)
    fig_state = create_state_diagram(chmm, x, a)
    fig_graph = create_graph_plot(chmm, x, a)
    
    linker.fig_umap = fig_umap
    linker.fig_state = fig_state
    linker.fig_graph = fig_graph
    
    # Set up event handlers
    fig_umap.data[0].on_selection(
        lambda trace, points, state: on_umap_selection(
            trace, points, {'linker': linker}
        )
    )
    
    # Create controls
    controls = create_linking_controls(linker)
    
    # Layout
    return widgets.VBox([
        controls,
        widgets.HBox([fig_umap, fig_state]),
        fig_graph
    ])
```

## Advanced Features

### Brush Selection

```python
def setup_brush_selection(fig, linker):
    """Enable brush selection in 2D projections."""
    # Add selection box
    fig.update_layout(
        dragmode='select',
        selectdirection='h'  # or 'v' or 'diagonal'
    )
    
    def on_selection(trace, points, state):
        # Get points in selection box
        selected_indices = points.point_inds
        linker.select_by_umap_indices(selected_indices)
    
    fig.data[0].on_selection(on_selection)
```

### Cross-Filtering

```python
def create_cross_filter(linker, filter_type='state'):
    """Create cross-filtering controls."""
    if filter_type == 'state':
        # Filter by state ID
        state_filter = widgets.IntText(description='State ID:')
        
        def apply_filter(change):
            state_id = state_filter.value
            umap_indices = find_umap_indices_for_state(
                linker.state_mapping, state_id
            )
            linker.select_by_umap_indices(umap_indices)
        
        state_filter.observe(apply_filter, names='value')
        return state_filter
```

## Performance Optimization

### Efficient Updates

```python
def batch_update_highlights(linker, updates):
    """Batch multiple highlight updates for performance."""
    # Collect all updates
    umap_updates = []
    graph_updates = []
    state_updates = []
    
    # Apply all at once
    with linker.fig_umap.batch_update():
        for update in umap_updates:
            apply_update(linker.fig_umap, update)
```

### Data Subsampling

```python
def subsample_for_linking(umap_embedding, max_points=10000):
    """Subsample UMAP data if too large for interactive linking."""
    if len(umap_embedding) > max_points:
        indices = np.random.choice(
            len(umap_embedding),
            max_points,
            replace=False
        )
        return umap_embedding[indices], indices
    return umap_embedding, np.arange(len(umap_embedding))
```

## Reference Implementation

Combine patterns from:

- `cscg/notebooks/Fig_2g.ipynb` for UMAP visualization
- `cscg/notebooks/intro.ipynb` for graph visualization
- `cscg/notebooks/ext_data_fig_8_a_Transition_graph.ipynb` for state analysis

Create a unified notebook that links all three visualization types with synchronized selection and highlighting.
