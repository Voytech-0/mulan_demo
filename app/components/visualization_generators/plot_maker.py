import numpy as np
import matplotlib.pyplot as plt
import io
import base64

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from components.configs.feature_config import IMAGE_ONLY_DATASETS
from components.visualization_generators.plot_helpers import encode_img_as_str, match_shape, create_main_fig_dataframe, create_color_map


def empty_fig(title='No dataset selected'):
    return px.scatter(title=title)

def invisible_interactable_layer(x_min, x_max, y_min, y_max):
    x_min, x_max = x_min - 1, x_max + 1
    y_min, y_max = y_min - 1, y_max + 1
    dummy_x, dummy_y = np.meshgrid(
        np.linspace(x_min, x_max, 50),
        np.linspace(y_min, y_max, 50)
    )
    return go.Scatter(
        x=dummy_x.flatten(),
        y=dummy_y.flatten(),
        mode='markers',
        opacity=1e-6,
        showlegend=False
    )

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
            name='User drawn' if int(label) == -1 else f'Additional: {label}',
            customdata=df_subset[['point_index', 'label']].values,
            hovertemplate='Index: %{customdata[0]}<br>Label: %{customdata[1]}<extra></extra>',
            showlegend=True
        ))

def create_datapoint_image(data_point, size=(20, 20)):
    normalized = (data_point - data_point.min()) / (data_point.max() - data_point.min())
    side_length = int(np.sqrt(len(normalized)))
    img_data = normalized.reshape(side_length, side_length)
    img_str = encode_img_as_str(img_data, size)
    img_data_url = f"data:image/png;base64,{img_str}"
    return img_data_url

def figure_with_images(df, X, title, category_orders):
    max_images = 100
    if len(X) > max_images:
        step = len(X) // max_images
        indices_to_show = list(range(0, len(X), step))[:max_images]
    else:
        indices_to_show = list(range(len(X)))
    images = [create_datapoint_image(X[i], size=(15, 15)) for i in indices_to_show]
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
        marker=dict(size=15, sizeref=1, sizemin=5, sizemode='diameter'),
        hoverinfo="text",
        hovertemplate=None,
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

def create_figure(embedding, y, title, X=None, is_thumbnail=False, show_images=False, class_names=None, n_added=0, dataset_name=None, **kwargs):
    if embedding is not None and hasattr(embedding, 'shape') and len(embedding.shape) == 3:
        embedding = embedding[-1]
    data_frames = create_main_fig_dataframe(embedding, X, y, class_names, n_added)
    df = data_frames[0]
    color_map = create_color_map(y)
    category_orders = {'color': class_names} if class_names is not None else None
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
            color_discrete_map=color_map,
            **kwargs
        )
    if n_added > 0 and embedding.shape[0] == X.shape[0]:
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
        fig.add_trace(invisible_interactable_layer(df['x'].min(), df['x'].max(), df['y'].min(), df['y'].max()))
        fig.update_layout(margin=dict(l=5, r=5, t=50, b=5))
    return fig

def create_animated_figure(embedding, y, title, label_name, class_names=None, n_added=0, X=None):
    is_continuous = len(np.unique(y)) > 20
    n_frames = max(1, len(embedding))
    color_map = create_color_map(y)
    category_orders = {'color': class_names} if class_names is not None else None
    # Compute global min/max for all frames for fixed axes
    all_x = np.concatenate([emb[:, 0] for emb in embedding[:n_frames]])
    all_y = np.concatenate([emb[:, 1] for emb in embedding[:n_frames]])
    x_range = [float(all_x.min()), float(all_x.max())]
    y_range = [float(all_y.min()), float(all_y.max())]
    fig = create_figure(
        embedding[0], y, title, X=X, is_thumbnail=False, show_images=True,
        class_names=class_names, n_added=n_added, dataset_name=None, range_x=x_range, range_y=y_range,
    )
    frames = []
    for i in range(0, n_frames, 10):
        data_frames_i = create_main_fig_dataframe(embedding[i], X, y, class_names, n_added)
        dfi = data_frames_i[0]
        scatter = go.Scatter(
            x=dfi['x'],
            y=dfi['y'],
            mode='markers',
            marker=dict(color=dfi['color']),
            showlegend=True
        )
        frames.append(go.Frame(data=[scatter], name=str(i)))
    fig.frames = frames
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

def create_figure(embedding, y, title, X=None, is_thumbnail=False, show_images=False, class_names=None, n_added=0,
                  dataset_name=None, interactive_figure=False):
    if embedding is None:
        return empty_fig('Not available')


    if hasattr(embedding, 'shape') and len(embedding.shape) == 3:
        embedding = embedding[-1]
        print("Using last frame of TRIMAP embedding for visualization")

    data_frames = create_main_fig_dataframe(embedding, X, y, class_names, n_added)
    category_orders = {'color': class_names}

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
            category_orders=category_orders
        )

    # Additional points
    if n_added > 0 and embedding.shape[0] == X.shape[0]:
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
        if interactive_figure:
            print('adding interactive figure')
            df = data_frames[0]
            fig.add_trace(invisible_interactable_layer(df['x'].min(), df['x'].max(), df['y'].min(), df['y'].max()))
        fig.update_layout(
            margin=dict(l=5, r=5, t=50, b=5)
        )
    print('finished creating figure')
    return fig

def create_data_distribution_plot(data, class_names=None, color_map=None):
    class_counts = pd.Series(data.target).value_counts().sort_index()
    df = pd.DataFrame({
        'Class': class_counts.index,
        'Count': class_counts.values
    })
    if class_names is not None:
        df['ClassName'] = [str(class_names[i]) if 0 <= i < len(class_names) else str(i) for i in df['Class']]
    else:
        df['ClassName'] = df['Class'].astype(str)
    color_discrete_map = {str(k): v for k, v in color_map.items()} if color_map is not None else None
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
    if is_continuous is None:
        is_continuous = len(np.unique(y)) > 20
    df = pd.DataFrame(X[:, :3], columns=['x', 'y', 'z'])
    df['label'] = y
    color_discrete_map = {str(k): v for k, v in color_map.items()} if color_map is not None else None
    if is_continuous:
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='label', title=title)
    else:
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='label', title=title, color_discrete_map=color_discrete_map)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        margin=dict(l=0, r=0, t=30, b=30)
    )
    return fig