"""
MULAN Demo Application - Manifold Learning Visualizations

A Dash-based web application for interactive visualization of manifold learning algorithms
including TRIMAP, t-SNE, and UMAP on various datasets.
"""
import logging

from dash import Dash
import dash_bootstrap_components as dbc
import plotly.io as pio
from components.cache import cache

from components.layout import create_layout
from components.callback_router import register_callbacks

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

cache.init_app(app.server)
app.layout = create_layout(app)

register_callbacks(app)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
    
    
