import jax.random as random
from sklearn import datasets
from sklearn.manifold import TSNE
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import numpy as np
import threading

# Import TRIMAP from local package
from google_research_trimap.trimap import trimap

# Try to import UMAP, if available
try:
    import umap
    umap_available = True
except ImportError:
    umap_available = False

# List of available datasets from sklearn.datasets
DATASET_LOADERS = {
    "Digits": datasets.load_digits,
    "Iris": datasets.load_iris,
    "Wine": datasets.load_wine,
    "Breast Cancer": datasets.load_breast_cancer,
}

# Single global lock for all Numba/embedding-related computations
numba_global_lock = threading.Lock()

def get_dataset(name):
    with numba_global_lock:
        loader = DATASET_LOADERS[name]
        data = loader()
        X = data.data
        y = data.target
    return X, y

def compute_trimap(X, key):
    with numba_global_lock:
        emb = trimap.transform(key, X, distance='euclidean')
        # Convert JAX array to numpy array if needed
        if hasattr(emb, "shape") and not isinstance(emb, np.ndarray):
            emb = np.array(emb)
    return emb

def compute_tsne(X):
    with numba_global_lock:
        emb = TSNE(n_components=2, random_state=42).fit_transform(X)
        return np.array(emb)

def compute_umap(X):
    if not umap_available:
        return None
    with numba_global_lock:
        reducer = umap.UMAP(n_components=2, random_state=42)
        emb = reducer.fit_transform(X)
        return np.array(emb)

def create_figure(embedding, y, title, label_name):
    # print number of points
    print(f"There are {len(embedding)} points for {title}")
    # Ensure embedding is a numpy array
    if embedding is None or len(embedding) == 0 or embedding.shape[1] < 2:
        return px.scatter(title=f"{title} (no data)")
    embedding = np.array(embedding)
    return px.scatter(
        x=embedding[:, 0],
        y=embedding[:, 1],
        color=y.astype(str),
        title=title,
        labels={'x': 'Component 1', 'y': 'Component 2', 'color': label_name}
    )

# Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Manifold Learning Visualizations"),
    html.Label("Select Dataset:"),
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[{'label': name, 'value': name} for name in DATASET_LOADERS.keys()],
        value='Digits'
    ),
    html.Div(id='umap-warning', style={'color': 'red'}),
    html.Div([
        html.H2("TRIMAP"),
        dcc.Graph(id='trimap-graph')
    ]),
    html.Div([
        html.H2("t-SNE"),
        dcc.Graph(id='tsne-graph')
    ]),
    html.Div([
        html.H2("UMAP"),
        dcc.Graph(id='umap-graph')
    ]),
])

@app.callback(
    Output('trimap-graph', 'figure'),
    Output('tsne-graph', 'figure'),
    Output('umap-graph', 'figure'),
    Output('umap-warning', 'children'),
    Input('dataset-dropdown', 'value')
)
def update_graphs(dataset_name):
    X, y = get_dataset(dataset_name)
    label_name = "Label"
    key = random.PRNGKey(42)
    # TRIMAP
    trimap_emb = compute_trimap(X, key)
    trimap_fig = create_figure(trimap_emb, y, f"TRIMAP Embedding of {dataset_name}", label_name)
    # t-SNE
    tsne_emb = compute_tsne(X)
    tsne_fig = create_figure(tsne_emb, y, f"t-SNE Embedding of {dataset_name}", label_name)
    # UMAP
    if umap_available:
        umap_emb = compute_umap(X)
        umap_fig = create_figure(umap_emb, y, f"UMAP Embedding of {dataset_name}", label_name)
        umap_warning = ""
    else:
        umap_fig = px.scatter(title="UMAP not available")
        umap_warning = "UMAP is not installed. Run 'pip install umap-learn' to enable UMAP visualizations."
    return trimap_fig, tsne_fig, umap_fig, umap_warning

if __name__ == "__main__":
    app.run(debug=True)
