import numpy as np
import matplotlib.pyplot as plt
import io
import base64

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from components.configs.feature_config import IMAGE_ONLY_DATASETS
from components.visualization_generators.plot_helpers import encode_img_as_str, match_shape, create_main_fig_dataframe


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


def add_new_data_to_fig(fig, df):
    for label in df['label'].unique():
        df_subset = df[df['label'] == label]
        fig.add_trace(go.Scatter(
            x=df_subset['x'],
            y=df_subset['y'],
            mode='markers',
            marker=dict(
                symbol='circle' if int(label) == -1 else 'x',
                size=40 if int(label) == -1 else 20,
                color=df_subset['color'],
                line=dict(width=2, color='black')
            ),
            name=f'Additional: {label}',
            customdata=df_subset[['point_index', 'label']].values,
            hovertemplate='Index: %{customdata[0]}<br>Label: %{customdata[1]}<extra></extra>',
            showlegend=True
        ))

_image_cache = {}


def create_datapoint_image(data_point, size=(20, 20)):
    normalized = (data_point - data_point.min()) / (data_point.max() - data_point.min())
    side_length = int(np.sqrt(len(normalized)))
    img_data = normalized.reshape(side_length, side_length)
    img_str = encode_img_as_str(img_data, size)
    img_data_url = f"data:image/png;base64,{img_str}"
    return img_data_url


def create_animated_figure(embedding, y, title, label_name):
    is_contiunuous = len(np.unique(y)) > 20
    n_frames = max(1, len(embedding))
    frames = []
    # Use y for all frames (assume y is static)
    point_indices = np.arange(len(y))
    # Initial frame
    if not is_contiunuous:
        df0 = pd.DataFrame({
            'x': embedding[0][:, 0],
            'y': embedding[0][:, 1],
            'color': y.astype(str),
            'color_num': y.astype(int) if np.issubdtype(y.dtype, np.integer) else pd.factorize(y)[0],
            'point_index': point_indices,
            'label': y.astype(str)
        })
    else:
        df0 = pd.DataFrame({
            'x': embedding[0][:, 0],
            'y': embedding[0][:, 1],
            'color': y,
            'label': y.astype(str)
        })
    # Use a consistent color palette for both px.scatter and go.Scatter
    color_palette = px.colors.qualitative.Plotly
    c_min = np.min(y)
    c_max = np.max(y)

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
        custom_data=['label'],
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
            'color': y,
            'label': y
        })
        scatter = go.Scatter(
            x=dfi['x'],
            y=dfi['y'],
            mode='markers',
            marker=dict(
                color=dfi['color'],
                colorscale=color_palette,
                cmin=c_min,
                cmax=c_max
            ),
            showlegend=False
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

def figure_with_images(df, X, title, category_orders):
    max_images = 100
    if len(X) > max_images:
        step = len(X) // max_images
        indices_to_show = list(range(0, len(X), step))[:max_images]
        print(
            f"Showing {len(indices_to_show)} images out of {len(X)} total points ({len(indices_to_show) / len(df) * 100:.1f}%)")
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

    # hover_texts = []
    # for i in df['point_index']:
    #     img_str = create_datapoint_image(X[i], size=(30, 30))
    #     hover_html = f"<img src='{img_str}' width='50' height='50'><br>Class: {df.iloc[i]['label']}"
    #     hover_texts.append(hover_html)

    fig.update_traces(
        marker=dict(
            size=15,
            sizeref=1,
            sizemin=5,
            sizemode='diameter'
        ),
        hoverinfo="text",
        hovertemplate=None,
        # text=hover_texts,
        selector=dict(type='scatter')
    )
    for i, img_str in enumerate(images):
        x, y = df.iloc[indices_to_show[i]]['x'], df.iloc[indices_to_show[i]]['y']
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
    return fig

def create_figure(embedding, y, title, X=None, is_thumbnail=False, show_images=False, class_names=None, n_added=0, dataset_name=None):
    if embedding is not None and hasattr(embedding, 'shape') and len(embedding.shape) == 3:
        embedding = embedding[-1]
        print("Using last frame of TRIMAP embedding for visualization")

    data_frames = create_main_fig_dataframe(embedding, X, y, class_names, n_added)
    category_orders = {'color': class_names}
    color_map = create_color_map(y)
    # Check if we should show images and if we have image data
    df = data_frames[0]
    if show_images and dataset_name in IMAGE_ONLY_DATASETS:
        fig = figure_with_images(df, X, title, category_orders)
    else:
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='label',
            custom_data=['point_index', 'label'],
            title=title,
            labels={'x': 'Component 1', 'y': 'Component 2', 'label': 'Class'},
            category_orders=category_orders,
            color_discrete_map=color_map
        )

    # Additional points
    if n_added > 0 and embedding is not None and X is not None and embedding.shape[0] != X.shape[0]:
        add_new_data_to_fig(fig, data_frames[1])

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
    print('finished creating figure')
    return fig

def create_color_map(y):
    unique_labels = np.sort(np.unique(y))
    color_seq = px.colors.qualitative.Plotly
    color_map = {str(label): color_seq[i % len(color_seq)] for i, label in enumerate(unique_labels)}
    return color_map

def create_data_distribution_plot(data, class_names=None, color_map=None):
    class_counts = pd.Series(data.target).value_counts().sort_index()

    df = pd.DataFrame({
        'Class': class_counts.index,
        'Count': class_counts.values
    })

    # If class_names are provided, map class indices to names for x-axis
    if class_names is not None:
        df['ClassName'] = [str(class_names[i]) if 0 <= i < len(class_names) else str(i) for i in df['Class']]
    else:
        df['ClassName'] = df['Class'].astype(str)

    # If color_map is provided, ensure keys are str and use as color_discrete_map
    color_discrete_map = None
    if color_map is not None:
        color_discrete_map = {str(k): v for k, v in color_map.items()}

    fig = px.bar(
        df,
        x='ClassName',
        y='Count',
        color='ClassName',
        labels={'ClassName': 'Class', 'Count': 'Sample Count'},
        color_discrete_map=color_discrete_map
    )

    fig.update_layout(
        title_font_size=16,
        font=dict(size=14),
        xaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=12),
            tickmode='linear',
            tick0=0,
            dtick=1
        ),
        yaxis=dict(
            title_font=dict(size=14),
            tickfont=dict(size=12)
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        margin=dict(l=0, r=0, t=30, b=30)
    )
    return fig

def create_3d_plot(X, y, title, class_names=None, color_map=None, is_continuous=None):
    import pandas as pd
    import numpy as np
    import plotly.express as px

    # Determine if the dataset should be continuous or categorical
    if is_continuous is None:
        is_continuous = len(np.unique(y)) > 20

    df = pd.DataFrame(X[:, :3], columns=['x', 'y', 'z'])
    if is_continuous:
        df['label'] = y
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='label', title=title)
    else:
        # y is already a list of string labels matching color_map keys
        df['label'] = y
        if color_map is not None:
            color_discrete_map = {str(k): v for k, v in color_map.items()}
            fig = px.scatter_3d(df, x='x', y='y', z='z', color='label', title=title, color_discrete_map=color_discrete_map)
        else:
            fig = px.scatter_3d(df, x='x', y='y', z='z', color='label', title=title)

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        margin=dict(l=0, r=0, t=30, b=30)
    )
    return fig