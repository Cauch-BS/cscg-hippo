---
name: umap-3d-visualization
description: Create interactive 3D UMAP visualizations from zarr cache neural data using Plotly. Use when working with vr2p ExperimentData, computing UMAP embeddings, creating 3D scatter plots with position markers, trial filtering, or interactive exploration of neural state spaces.
---

# 3D UMAP Visualization from Zarr Cache

Create interactive 3D UMAP visualizations of neural activity data loaded from zarr cache files.

## Quick Start

Load data from zarr cache and create interactive 3D UMAP plot:

```python
import vr2p
import umap
import plotly.graph_objects as go
import ipywidgets as widgets

# Load data
path = '/path/to/data.zarr'
data = vr2p.ExperimentData(path)

# Extract spike data (see Fig_2g.ipynb for full preprocessing)
spks_big_array = ...  # Shape: (n_neurons, n_timepoints)

# Compute 3D UMAP
umap_model = umap.UMAP(n_neighbors=100, n_components=3, min_dist=0.1,
                       metric='correlation', random_state=42)
umap_embedding = umap_model.fit_transform(spks_big_array.T)

# Create interactive plot
fig = go.FigureWidget(data=[go.Scatter3d(
    x=umap_embedding[:, 0],
    y=umap_embedding[:, 1],
    z=umap_embedding[:, 2],
    mode='markers',
    marker=dict(size=1.6, color=colors, opacity=0.8),
    customdata=customdata,
    hovertemplate="<br>".join([
        "Trial Type: %{customdata[2]}",
        "Position: %{customdata[3]}",
        "Area: %{customdata[0]}",
        "Trial Number: %{customdata[1]}",
        "Set: %{customdata[4]}",
    ])
)])
```

## Data Loading

### From Zarr Cache

```python
import vr2p

# Load experiment data
data = vr2p.ExperimentData('/path/to/data.zarr')

# Access multi-session data
F_big = []  # Fluorescence signals
spks_big = []  # Spike data
for session_id in range(len(data.vr)):
    vr = data.vr[session_id]
    # Extract relevant frames (see Fig_2g.ipynb for filtering logic)
    selected_frames = ...
    F_big.append(data.signals.multi_session.Fns[session_id][:, selected_frames])
    spks_big.append(data.signals.multi_session.spks[session_id][:, selected_frames])

# Concatenate across sessions
spks_big_array = np.hstack(spks_big)
```

### Position Markers

Define position markers for color coding:

```python
markers = [
    {'name': 'Track', 'color': '#808080', 'position': 0.9},
    {'name': 'Indicator-Near', 'color': '#FBB4B9', 'position': 0.85},
    {'name': 'R1-Near', 'color': '#F768A1', 'position': 0.8},
    {'name': 'R2-Near', 'color': '#C51B8A', 'position': 0.75},
    {'name': 'Indicator-Far', 'color': '#A8D8A7', 'position': 0.7},
    {'name': 'R1-Far', 'color': '#41AE76', 'position': 0.65},
    {'name': 'R2-Far', 'color': '#006D2C', 'position': 0.6},
    {'name': 'Teleportation', 'color': '#000000', 'position': 0.55},
]
```

## UMAP Computation

### Basic 3D UMAP

```python
import umap

umap_model = umap.UMAP(
    n_neighbors=100,      # Local neighborhood size
    n_components=3,       # 3D embedding
    min_dist=0.1,         # Minimum distance between points
    metric='correlation',  # Distance metric
    random_state=42       # Reproducibility
)

umap_embedding = umap_model.fit_transform(spks_big_array.T)
# Shape: (n_timepoints, 3)
```

### Saving/Loading Embeddings

```python
# Save
np.save('umap_embedding.npy', umap_embedding)

# Load
umap_embedding = np.load('umap_embedding.npy')
```

## Interactive Plotly Visualization

### Basic 3D Scatter

```python
import plotly.graph_objects as go

fig = go.FigureWidget(data=[go.Scatter3d(
    x=umap_embedding[:, 0],
    y=umap_embedding[:, 1],
    z=umap_embedding[:, 2],
    mode='markers',
    marker=dict(
        size=1.6,
        color=color_array,  # Array of colors for each point
        opacity=0.8
    ),
    customdata=customdata,  # Metadata for hover
    hovertemplate="Area: %{customdata[0]}<br>Trial: %{customdata[1]}"
)])

fig.update_layout(
    scene=dict(
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        zaxis_title="UMAP 3"
    ),
    width=1200,
    height=650
)
```

### Color Schemes

**By Position Marker**:

```python
# Assign colors based on position_marker column
for marker in markers:
    selected_position.loc[
        selected_position['position_marker'] == marker['name'],
        'area-color'
    ] = marker['color']
```

**By Position (Gradient)**:

```python
from matplotlib import cm, colors

norm = colors.Normalize(vmin=0, vmax=230)
for reward_id in [1, 2]:
    cmap = cm.get_cmap('Blues' if reward_id == 1 else 'YlOrBr')
    ind = selected_position['reward_id'] == reward_id
    selected_position.loc[ind, 'position-color'] = list(
        map(colors.rgb2hex, cmap(norm(selected_position.loc[ind, 'position'])))
    )
```

**By Trial Number**:

```python
norm = colors.Normalize(vmin=0, vmax=100)
trial_number_list = norm(selected_position['trial_number']).astype(float)
selected_position['trial-color'] = list(
    map(colors.rgb2hex, cmap(trial_number_list))
)
```

## Interactive Controls

### Camera Controls

```python
import ipywidgets as widgets

# Camera position controls
up_x = widgets.FloatText(value=0, description='up_x')
up_y = widgets.FloatText(value=0, description='up_y')
up_z = widgets.FloatText(value=1, description='up_z')
eye_x = widgets.FloatText(value=-1.5, description='eye_x')
eye_y = widgets.FloatText(value=1.5, description='eye_y')
eye_z = widgets.FloatText(value=1.5, description='eye_z')

def update_view(b):
    fig.update_layout(scene_camera=dict(
        up=dict(x=up_x.value, y=up_y.value, z=up_z.value),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=eye_x.value, y=eye_y.value, z=eye_z.value)
    ))

update_button = widgets.Button(description='Update View')
update_button.on_click(update_view)
```

### Trial Selection

```python
trial_selector = widgets.IntSlider(
    value=selected_position.trial_number.min(),
    min=selected_position.trial_number.min(),
    max=selected_position.trial_number.max(),
    step=1,
    description='Trial Number'
)

trial_use_selected = widgets.Checkbox(
    value=False,
    description='Highlight selected'
)

def update_trial_highlight(change):
    if trial_use_selected.value:
        # Filter and highlight selected trial
        mask = selected_position['trial_number'] == trial_selector.value
        # Update marker colors/sizes
        pass

trial_selector.observe(update_trial_highlight, names='value')
```

### Color Scheme Selection

```python
color_options = [
    'Trial Type - Areas',
    'Trial Type - Position',
    'Trial Type - Trial Number',
    'Cue Sets'
]

color_scheme_selector = widgets.Dropdown(
    options=color_options,
    value=color_options[0],
    description='Color:'
)

def update_colors(change):
    scheme = color_scheme_selector.value
    if scheme == 'Trial Type - Areas':
        colors = selected_position['area-color']
    elif scheme == 'Trial Type - Position':
        colors = selected_position['position-color']
    # ... update figure colors
    fig.data[0].marker.color = colors

color_scheme_selector.observe(update_colors, names='value')
```

## Export Functions

### Save PDF/PNG

```python
def save_pdf(b):
    fig.write_image(
        "umap_3d.pdf",
        engine="kaleido",
        width=600,
        height=600,
        scale=10
    )

def save_png(b):
    fig.write_image(
        "umap_3d.png",
        engine="kaleido",
        width=600,
        height=600,
        scale=10
    )

save_pdf_button = widgets.Button(description='Save PDF')
save_pdf_button.on_click(save_pdf)
```

### Save Camera Coordinates

```python
def save_coordinates(b):
    current_coordinates = np.array([
        [up_x.value, up_y.value, up_z.value],
        [center_x.value, center_y.value, center_z.value],
        [eye_x.value, eye_y.value, eye_z.value]
    ])
    np.save('camera_coordinates.npy', current_coordinates)
```

## Complete UI Layout

```python
ui = widgets.VBox([
    color_scheme_container,
    trial_container,
    view_params_container,
    save_coordinates_button,
    fig
])
display(ui)
```

## Performance Considerations

- For large datasets, consider subsampling before UMAP computation
- Use efficient data structures (numpy arrays, zarr) for memory management
- Cache UMAP embeddings to avoid recomputation
- Use `FigureWidget` for interactive updates without full redraws

## Reference Implementation

See `cscg/notebooks/Fig_2g.ipynb` for a complete working example with:

- Multi-session data loading
- Position marker assignment
- Multiple color schemes
- Full interactive controls
- Export functionality
