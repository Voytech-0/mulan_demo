from dash import Dash
import dash_bootstrap_components as dbc
from components.layout import create_layout
from components.callbacks import register_callbacks
import plotly.io as pio

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, '/assets/custom.css'])
app.title = "Manifold Learning Visualizations"
pio.templates.default = "plotly_dark"

app.layout = create_layout(app)
app.config.suppress_callback_exceptions = True
register_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True)