import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import base64
from io import BytesIO

from .features import dataset_shape


def encode_img_as_str(data_point, dataset_name):
    img_shape = dataset_shape(dataset_name)
    plt.figure(figsize=(4, 4))
    plt.imshow(data_point.reshape(img_shape), cmap='gray')
    plt.axis('off')

    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return img_str

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
