"""
MULAN Demo Application - Manifold Learning Visualizations

A Dash-based web application for interactive visualization of manifold learning algorithms
including TRIMAP, t-SNE, and UMAP on various datasets.
"""

from dash import Dash
import dash_bootstrap_components as dbc
import plotly.io as pio

from components.layout import create_layout
from components.callbacks import register_callbacks

# Configure Plotly template
pio.templates.default = "plotly_dark"

# Initialize the Dash application
app = Dash(
    __name__, 
    external_stylesheets=[dbc.themes.DARKLY, '/assets/custom.css']
)
app.title = "Manifold Learning Visualizations"
app.config.suppress_callback_exceptions = True

# Set up layout and callbacks
app.layout = create_layout(app)
register_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True)
    
    
