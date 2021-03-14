# Created at 2020-05-27
# Summary: isosurface plot

import plotly.graph_objects as go
import numpy as np

# https://plotly.com/python/3d-isosurface-plots/#basic-isosurface
X, Y, Z = np.mgrid[-5:5:50j, -5:5:50j, -5:5:50j]

# values =  X * X  + Y * Y + Z * Z #
values = (X-Y)**2 + (Y-Z)**2 + (X-Z)**2 # X * X * 0.5 + Y * Y + Z * Z * 2

fig = go.Figure(data=go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    isomin=4,
    isomax=50,
    surface=dict(count=3, fill=0.5, pattern='odd'),
    showscale=True, # remove colorbar
    caps=dict(x_show=True, y_show=True),
    ))

fig.update_layout(
    margin=dict(t=0, l=0, b=0), # tight layout
    scene_camera_eye=dict(x=1.86, y=0.61, z=0.98))
fig.show()