import pandas as pd
import numpy as np
from sklearn import cluster
import plotly.io as pio
pio.renderers.default = "browser"

data = pd.read_csv("Spotify-2000.csv")
data = data.drop("Index",axis=1)
print(data.corr())
#print(data.head())

data2 = data[["Beats Per Minute (BPM)", "Loudness (dB)",
              "Liveness", "Valence", "Acousticness",
              "Speechiness"]]

from sklearn.preprocessing import MinMaxScaler

for i in data.columns:
    MinMaxScaler(i)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(data2)

data["Music Segments"] = clusters
MinMaxScaler(data["Music Segments"])
data["Music Segments"] = data["Music Segments"].map({1: "Cluster 1", 2:
    "Cluster 2", 3: "Cluster 3", 4: "Cluster 4", 5: "Cluster 5",
    6: "Cluster 6", 7: "Cluster 7", 8: "Cluster 8",
    9: "Cluster 9", 10: "Cluster 10"})

print(data.head())

import plotly.graph_objects as go

PLOT = go.Figure()
for i in list(data["Music Segments"].unique()):
    PLOT.add_trace(go.Scatter3d(x=data[data["Music Segments"] == i]['Beats Per Minute (BPM)'],
                                y=data[data["Music Segments"] == i]['Energy'],
                                z=data[data["Music Segments"] == i]['Danceability'],
                                mode='markers', marker_size=6, marker_line_width=1,
                                name=str(i)))
PLOT.update_traces(hovertemplate='Beats Per Minute (BPM): %{x} <br>Energy: %{y} <br>Danceability: %{z}')

PLOT.update_layout(width=800, height=800, autosize=True, showlegend=True,
                   scene=dict(xaxis=dict(title='Beats Per Minute (BPM)', titlefont_color='black'),
                              yaxis=dict(title='Energy', titlefont_color='black'),
                              zaxis=dict(title='Danceability', titlefont_color='black')),
                   font=dict(family="Gilroy", color='black', size=12))

PLOT.show()
