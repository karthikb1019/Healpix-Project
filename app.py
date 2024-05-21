import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objects as go

def create_healpix(nside):
    """
    Create HEALPix pixel centers for a given nside.
    
    Parameters:
    nside (int): Resolution parameter. The total number of pixels is 12 * nside^2.
    
    Returns:
    theta (np.ndarray): Colatitudes of the pixels.
    phi (np.ndarray): Longitudes of the pixels.
    """
    npix = 12 * nside ** 2
    theta = np.zeros(npix)
    phi = np.zeros(npix)
    
    for i in range(npix):
        z = 1 - (2 * i + 1) / npix
        theta[i] = np.arccos(z)
        phi[i] = (np.sqrt(2) * np.pi * i) % (2 * np.pi)
    
    return theta, phi

def create_sphere():
    """
    Create a mesh grid for a sphere.
    
    Returns:
    x (np.ndarray): X-coordinates of the sphere.
    y (np.ndarray): Y-coordinates of the sphere.
    z (np.ndarray): Z-coordinates of the sphere.
    """
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

def plot_healpix_grid(nside, color):
    """
    Plot HEALPix grid points on a sphere.
    
    Parameters:
    nside (int): Resolution parameter.
    color (str): Color of the points.
    
    Returns:
    dict: Scatter3d trace.
    """
    theta, phi = create_healpix(nside)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    text = [f'Point {i}<br>Theta: {theta[i]:.2f}<br>Phi: {phi[i]:.2f}' for i in range(len(theta))]
    
    return go.Scatter3d(
        x=x, y=y, z=z, mode='markers', marker=dict(size=4, color=color), text=text, hoverinfo='text', name=f'{12 * nside ** 2} pixels'
    )

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Interactive HEALPix Visualization", style={'text-align': 'center'}),
    html.Div([
        html.P("HEALPix is an acronym for Hierarchical Equal Area isoLatitude Pixelization of a sphere. It is a method of pixelizing a spherical surface such that each pixel covers the same surface area. This technique is widely used in astronomy, particularly in mapping the cosmic microwave background (CMB) radiation."),
        html.P("In this visualization, you can see the HEALPix grid at different resolutions. Use the slider to change the resolution and interact with the sphere to see how the grid points are distributed."),
        html.P("The green sphere represents the lowest resolution with 12 pixels, yellow has 48 pixels, red has 192 pixels, and blue has 768 pixels. The pixel centers are located on rings of constant latitude, which are equally spaced in azimuthal direction."),
    ], style={'max-width': '800px', 'margin': 'auto'}),
    
    dcc.Graph(id='sphere-graph', style={'height': '70vh'}),
    
    html.Div([
        html.Label('Select Resolution:'),
        dcc.Slider(
            id='nside-slider',
            min=1,
            max=8,
            value=1,
            marks={1: '12', 2: '48', 4: '192', 8: '768'},
            step=None
        )
    ], style={'width': '60%', 'margin': 'auto'}),
    
    html.Footer("Developed by Your Name", style={'text-align': 'center', 'margin-top': '20px', 'padding': '10px', 'background': '#f0f0f0'})
], style={'font-family': 'Arial, sans-serif', 'margin': '0', 'padding': '0', 'box-sizing': 'border-box'})

@app.callback(
    Output('sphere-graph', 'figure'),
    [Input('nside-slider', 'value')]
)
def update_graph(nside):
    colors = ['green', 'yellow', 'red', 'blue']
    
    fig = go.Figure()
    
    # Plot the sphere
    x, y, z = create_sphere()
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='Blues', opacity=0.3, showscale=False))
    
    # Plot HEALPix grid
    color_index = {1: 0, 2: 1, 4: 2, 8: 3}
    color = colors[color_index.get(nside, 0)]
    
    fig.add_trace(plot_healpix_grid(nside, color))
    
    # Set plot layout
    fig.update_layout(
        title='Interactive HEALPix Grid on a Sphere',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
