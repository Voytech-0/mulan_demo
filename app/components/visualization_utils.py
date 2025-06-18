"""
Visualization utilities for the MULAN demo app.
Handles creation of plots, figures, and image processing.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import base64
from .settings import IMAGE_FIGURE_SIZE


def create_datapoint_image(data_point, size=(20, 20)):
    """
    Create an image from a data point (for image datasets).
    
    Args:
        data_point: 1D array of pixel values
        size: Tuple of (width, height) for the image
        
    Returns:
        Base64 encoded image string
    """
    try:
        # Reshape the data point to 2D
        if len(data_point) == 784:  # MNIST/Fashion MNIST size
            img_array = data_point.reshape(28, 28)
        elif len(data_point) == 64:  # Digits size
            img_array = data_point.reshape(8, 8)
        else:
            # Try to make it square, or use the original size
            side_length = int(np.sqrt(len(data_point)))
            if side_length * side_length == len(data_point):
                img_array = data_point.reshape(side_length, side_length)
            else:
                # Pad or truncate to make it square
                side_length = int(np.sqrt(len(data_point)))
                padded = np.zeros(side_length * side_length)
                padded[:len(data_point)] = data_point
                img_array = padded.reshape(side_length, side_length)
        
        # Normalize to 0-255 range
        img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
        
        # Create PIL image
        img = Image.fromarray(img_array, mode='L')
        img = img.resize(size, Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        print(f"Error creating image from data point: {e}")
        return ""


def create_figure(embedding, y, title, label_name, X=None, is_thumbnail=False, show_images=False, class_names=None):
    """
    Create a Plotly figure for embedding visualization.
    
    Args:
        embedding: 2D embedding coordinates
        y: Target labels
        title: Figure title
        label_name: Name for the label axis
        X: Original data (for image display)
        is_thumbnail: Whether this is a thumbnail figure
        show_images: Whether to show images instead of points
        
    Returns:
        Plotly figure object
    """
    if embedding is None or len(embedding) == 0:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title=title,
            xaxis_title="X",
            yaxis_title="Y",
            showlegend=False
        )
        return fig
    
    # Create scatter plot
    if show_images and X is not None:
        # Create figure with images
        fig = go.Figure()
        
        # Sample points for performance (max 100 points for image display)
        n_points = min(100, len(embedding))
        indices = np.random.choice(len(embedding), n_points, replace=False)
        
        for i, idx in enumerate(indices):
            img_str = create_datapoint_image(X[idx])
            if img_str:
                fig.add_layout_image(
                    dict(
                        source=img_str,
                        x=embedding[idx, 0],
                        y=embedding[idx, 1],
                        xref="x",
                        yref="y",
                        sizex=0.5,
                        sizey=0.5,
                        sizing="scaled",
                        layer="above"
                    )
                )
    else:
        # Create regular scatter plot
        unique_labels = np.unique(y)
        colors = px.colors.qualitative.Set3[:len(unique_labels)]
        
        fig = go.Figure()
        
        for i, label in enumerate(unique_labels):
            mask = y == label
            display_label = class_names[label]
            fig.add_trace(go.Scatter(
                x=embedding[mask, 0],
                y=embedding[mask, 1],
                mode='markers',
                name=display_label,
                marker=dict(
                    size=6 if is_thumbnail else 8,
                    color=colors[i % len(colors)],
                    opacity=0.7
                ),
                hovertemplate=f'<b>{label_name}:</b> %{{text}}<br>' +
                             '<b>X:</b> %{x}<br>' +
                             '<b>Y:</b> %{y}<extra></extra>',
                text=[display_label] * np.sum(mask)
            ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
        showlegend=not is_thumbnail,
        margin=dict(l=20, r=20, t=40, b=20),
        height=300 if is_thumbnail else 600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.1)',
        zeroline=False
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255,255,255,0.1)',
        zeroline=False
    )
    
    return fig


def create_metadata_table(dataset_name, data, point_index=None):
    """
    Create a metadata table for display.
    
    Args:
        dataset_name: Name of the dataset
        data: Dataset object
        point_index: Index of specific point (if None, show dataset info)
        
    Returns:
        HTML table component
    """
    from dash import html
    from .feature_config import DATASET_FEATURES, IMAGE_ONLY_DATASETS
    from .settings import TABLE_STYLE, CELL_STYLE, CELL_STYLE_RIGHT, EMPTY_METADATA_STYLE
    
    if dataset_name in IMAGE_ONLY_DATASETS:
        return html.Div("No metadata available for image datasets", style=EMPTY_METADATA_STYLE)
    
    if point_index is not None:
        # Show point-specific metadata
        features = DATASET_FEATURES.get(dataset_name, [])
        if not features:
            return html.Div("No features configured for this dataset", style=EMPTY_METADATA_STYLE)
        
        rows = []
        for feature in features:
            if feature in data.feature_names:
                idx = data.feature_names.index(feature)
                value = data.data[point_index, idx]
                rows.append(html.Tr([
                    html.Td(feature, style=CELL_STYLE),
                    html.Td(f"{value:.4f}", style=CELL_STYLE_RIGHT)
                ]))
        
        if not rows:
            return html.Div("No metadata available for this point", style=EMPTY_METADATA_STYLE)
        
        return html.Table([
            html.Thead(html.Tr([
                html.Th("Feature", style=CELL_STYLE),
                html.Th("Value", style=CELL_STYLE_RIGHT)
            ])),
            html.Tbody(rows)
        ], style=TABLE_STYLE)
    else:
        # Show dataset-level information
        rows = [
            html.Tr([html.Td("Dataset", style=CELL_STYLE), html.Td(dataset_name, style=CELL_STYLE_RIGHT)]),
            html.Tr([html.Td("Samples", style=CELL_STYLE), html.Td(str(data.data.shape[0]), style=CELL_STYLE_RIGHT)]),
            html.Tr([html.Td("Features", style=CELL_STYLE), html.Td(str(data.data.shape[1]), style=CELL_STYLE_RIGHT)]),
            html.Tr([html.Td("Classes", style=CELL_STYLE), html.Td(str(len(data.target_names)), style=CELL_STYLE_RIGHT)]),
        ]
        
        return html.Table([
            html.Thead(html.Tr([
                html.Th("Property", style=CELL_STYLE),
                html.Th("Value", style=CELL_STYLE_RIGHT)
            ])),
            html.Tbody(rows)
        ], style=TABLE_STYLE) 