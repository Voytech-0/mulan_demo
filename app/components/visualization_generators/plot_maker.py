import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from components.visualization_generators.plot_helpers import encode_img_as_str

def empty_fig(title='No dataset selected'):
    return px.scatter(title=title)

def invisible_interactable_layer(x_min, x_max, y_min, y_max):
    x_min, x_max = x_min - 1, x_max + 1
    y_min, y_max = y_min - 1, y_max + 1

    # Create a meshgrid of invisible points
    dummy_x, dummy_y = np.meshgrid(
        np.linspace(x_min, x_max, 50),
        np.linspace(y_min, y_max, 50)
    )

    # Add invisible dummy scatter layer
    layer = go.Scatter(
        x=dummy_x.flatten(),
        y=dummy_y.flatten(),
        mode='markers',
        opacity=1e-6,
        # hoverinfo='skip',
        showlegend=False
    )
    return layer


def add_new_data_to_fig(fig, df, color_map):
    for label in df['label'].unique():
        df_subset = df[df['label'] == label]
        fig.add_trace(go.Scatter(
            x=df_subset['x'],
            y=df_subset['y'],
            mode='markers',
            marker=dict(
                symbol='circle' if label == '-1.0' else 'x',
                size=20 if label == '-1.0' else 10,
                color=color_map[label],
                line=dict(width=2, color='black')
            ),
            name=f'Additional: {label}',
            customdata=df_subset[['point_index', 'label']].values,
            hovertemplate='Index: %{customdata[0]}<br>Label: %{customdata[1]}<extra></extra>',
            showlegend=True
        ))

_image_cache = {}


def create_datapoint_image(data_point, size=(20, 20)):
    """Create a small image representation of a datapoint."""
    # Create a cache key based on the data point and size
    cache_key = (hash(data_point.tobytes()), size)

    # Check if we have this image cached
    if cache_key in _image_cache:
        return _image_cache[cache_key]

    # Normalize the data point to 0-1 range
    normalized = (data_point - data_point.min()) / (data_point.max() - data_point.min())

    img_data_url = encode_img_as_str(normalized)

    # Cache the result
    _image_cache[cache_key] = img_data_url

    # Limit cache size to prevent memory issues
    if len(_image_cache) > 1000:
        # Remove oldest entries (simple FIFO)
        oldest_keys = list(_image_cache.keys())[:100]
        for key in oldest_keys:
            del _image_cache[key]

    return img_data_url


def create_animated_figure(embedding, y, title, label_name):
    n_frames = max(1, len(embedding))
    frames = []
    # Use y for all frames (assume y is static)
    point_indices = np.arange(len(y))
    # Initial frame
    df0 = pd.DataFrame({
        'x': embedding[0][:, 0],
        'y': embedding[0][:, 1],
        'color': y.astype(str),
        'color_num': y.astype(int) if np.issubdtype(y.dtype, np.integer) else pd.factorize(y)[0],
        'point_index': point_indices,
        'label': y.astype(str)
    })
    # Use a consistent color palette for both px.scatter and go.Scatter
    color_palette = px.colors.qualitative.Plotly

    # Compute global min/max for all frames for fixed axes
    all_x = np.concatenate([emb[:, 0] for emb in embedding[:n_frames]])
    all_y = np.concatenate([emb[:, 1] for emb in embedding[:n_frames]])
    x_range = [float(all_x.min()), float(all_x.max())]
    y_range = [float(all_y.min()), float(all_y.max())]

    fig = px.scatter(
        df0,
        x='x',
        y='y',
        color='color',
        custom_data=['point_index', 'label'],
        title=title,
        labels={'x': 'Component 1', 'y': 'Component 2', 'color': label_name},
        color_discrete_sequence=color_palette,
        range_x=x_range,
        range_y=y_range
    )
    # Build frames
    for i in range(0, n_frames, 10):
        dfi = pd.DataFrame({
            'x': embedding[i][:, 0],
            'y': embedding[i][:, 1],
            'color': y.astype(str),
            'color_num': y.astype(int) if np.issubdtype(y.dtype, np.integer) else pd.factorize(y)[0],
            'point_index': point_indices,
            'label': y.astype(str)
        })
        scatter = go.Scatter(
            x=dfi['x'],
            y=dfi['y'],
            mode='markers',
            marker=dict(
                color=dfi['color_num'],
                colorscale=color_palette,
                cmin=0,
                cmax=len(color_palette) - 1
            ),
            customdata=np.stack([dfi['point_index'], dfi['label']], axis=-1),
            showlegend=False,
            hovertemplate="Class: %{customdata[1]}<br>Index: %{customdata[0]}<br>X: %{x}<br>Y: %{y}<extra></extra>"
        )
        frames.append(go.Frame(data=[scatter], name=str(i)))
    fig.frames = frames

    # Add animation slider
    sliders = [{
        "steps": [
            {
                "args": [
                    [str(k)],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }
                ],
                "label": str(k),
                "method": "animate"
            } for k in range(0, n_frames, 10)
        ],
        "transition": {"duration": 0},
        "x": 0.1,
        "y": -0.15,
        "currentvalue": {"font": {"size": 14}, "prefix": "Frame: ", "visible": True, "xanchor": "center"},
        "len": 0.9
    }]
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": 50, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                        }
                    ],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }
                    ],
                    "label": "Pause",
                    "method": "animate",
                    "args": [
                        [],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": False},
                            "transition": {"duration": 0}
                        }
                    ]
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "showactive": False,
            "x": 0.1,
            "y": -0.15,
            "xanchor": "right",
            "yanchor": "top"
        }],
        sliders=sliders,
        margin=dict(l=5, r=5, t=50, b=5)
    )
    return fig

def create_figure(embedding, y, title, label_name, X=None, is_thumbnail=False, show_images=False, class_names=None, n_added=0):
    try:
        if embedding is None or len(embedding) == 0 or embedding.shape[-1] < 2:
            return px.scatter(title=f"{title} (no data)")
    except TypeError:
        return px.scatter(title=f"{title} (no data)")

    if len(embedding.shape) > 2:
        embedding = embedding[-1]

    # Create a list of customdata for each point, including the point index
    point_indices = np.arange(len(y))

    # If class_names is not provided, use unique values in y as strings
    if class_names is None:
        unique_classes = np.unique(y)
        class_names = [str(c) for c in unique_classes]

    # Map y to class names for legend
    y_int = y.astype(int)
    y_labels = [str(class_names[i]) if 0 <= i < len(class_names) else str(i) for i in y_int]

    unique_labels = pd.Series(y_labels).unique()
    color_seq = px.colors.qualitative.Plotly
    color_map = {label: color_seq[i % len(color_seq)] for i, label in enumerate(sorted(unique_labels))}
    n_original = len(y) - n_added
    sections = [slice(0, n_original)]

    visualize_added_samples = n_added > 0 and embedding.shape[0] == X.shape[0]
    if visualize_added_samples:
        sections.append(slice(n_original, None))

    data_frames = []
    for section in sections:
        df = pd.DataFrame({
            'x': embedding[section, 0],
            'y': embedding[section, 1],
            'point_index': point_indices[section],
            'label': y_labels[section]
        })
        df['color'] = df['label'].map(color_map)
        data_frames.append(df)

    # Set category order for consistent color mapping
    category_orders = {'color': class_names}

    # Check if we should show images and if we have image data
    df = data_frames[0]
    if show_images and X is not None and len(X) > 0:
        if len(X[0]) in [64, 784]:
            max_images = 100
            if len(X) > max_images:
                step = len(X) // max_images
                indices_to_show = list(range(0, len(X), step))[:max_images]
                print(f"Showing {len(indices_to_show)} images out of {len(X)} total points ({len(indices_to_show)/len(X)*100:.1f}%)")
            else:
                indices_to_show = list(range(len(X)))
                print(f"Showing all {len(indices_to_show)} images")
            images = []
            for i in indices_to_show:
                img_str = create_datapoint_image(X[i], size=(15, 15))
                images.append(img_str)
            fig = px.scatter(
                df,
                x='x',
                y='y',
                color='label',
                custom_data=['point_index', 'label'],
                title=title,
                labels={'x': 'Component 1', 'y': 'Component 2', 'label': 'Class'},
                category_orders=category_orders
            )
            fig.update_traces(
                marker=dict(
                    size=15,
                    sizeref=1,
                    sizemin=5,
                    sizemode='diameter'
                ),
                selector=dict(type='scatter')
            )
            if show_images and X is not None:
                hover_texts = []
                for i in df['point_index']:
                    img_str = create_datapoint_image(X[i], size=(30, 30))
                    hover_html = f"<img src='{img_str}' width='50' height='50'><br>Class: {df.iloc[i]['label']}"
                    hover_texts.append(hover_html)

            fig.update_traces(
                marker=dict(
                    size=15,
                    sizeref=1,
                    sizemin=5,
                    sizemode='diameter'
                ),
                hoverinfo="text",
                hovertemplate=None,
                text=hover_texts,
                selector=dict(type='scatter')
            )
            for i, (idx, img_str) in enumerate(zip(indices_to_show, images)):
                x, y = df.iloc[idx]['x'], df.iloc[idx]['y']
                x_range = df['x'].max() - df['x'].min()
                y_range = df['y'].max() - df['y'].min()
                base_size = max(x_range, y_range) * 0.04
                fig.add_layout_image(
                    dict(
                        source=img_str,
                        xref="x",
                        yref="y",
                        x=x,
                        y=y,
                        sizex=base_size,
                        sizey=base_size,
                        xanchor="center",
                        yanchor="middle"
                    )
                )
        else:
            fig = px.scatter(
                df,
                x='x',
                y='y',
                color='label',
                custom_data=['point_index', 'label'],
                title=title,
                labels={'x': 'Component 1', 'y': 'Component 2', 'label': 'Class'},
                category_orders=category_orders
            )
    else:
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='label',
            custom_data=['point_index', 'label'],
            title=title,
            labels={'x': 'Component 1', 'y': 'Component 2', 'label': 'Class'},
            category_orders=category_orders
        )


    # Additional points
    if visualize_added_samples:
        add_new_data_to_fig(fig, data_frames[1], color_map)

    if is_thumbnail:
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showticklabels=False, visible=False),
            yaxis=dict(showticklabels=False, visible=False),
            showlegend=False,
            hovermode=False
        )
    else:
        df = data_frames[0]
        fig.add_trace(invisible_interactable_layer(df['x'].min(), df['x'].max(), df['y'].min(), df['y'].max()))
        fig.update_layout(
            margin=dict(l=5, r=5, t=50, b=5)
        )
    return fig
