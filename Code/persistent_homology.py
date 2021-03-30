import numpy as np
import pandas as pd
# ^^^ pyforest auto-imports - don't write above this line
from scipy.spatial.distance import pdist, squareform
import gtda.homology as hl
from gtda.plotting import plot_diagram
import plotly.graph_objects as go



def generate_persistent_plot(subreddit, 
                             pickle_location='Download/Cleaned Data with Longformer/',
                             metric='euclidean',
                             range=[0,50],
                             sample=200):
    SUBREDDIT = subreddit
    df = pd.read_pickle(f'{pickle_location}{SUBREDDIT}.csv.pkl')
    # df = pd.read_pickle('Download/Cleaned Data with Longformer/Confessions.csv.pkl')
    df = df.sample(n=sample, replace=False)

    df = pd.DataFrame(np.concatenate(df['LF pooler output'].apply(lambda row: row.numpy()).to_numpy()))

    dis_matrix = squareform(pdist(df.to_numpy(), metric=metric))
    dis_matrix = dis_matrix.reshape(1,*(dis_matrix.shape))

    

    # represent data as a point cloud
    point_cloud = dis_matrix

    # define topological features to track
    homology_dimensions = [0, 1, 2]

    # define simplicial complex to construct
    persistence = hl.VietorisRipsPersistence(
        metric="precomputed", homology_dimensions=homology_dimensions
    )

    # calculate persistence diagram
    persistence_diagram = persistence.fit_transform(dis_matrix)


    fig = plot_diagram(persistence_diagram[0], 
                    plotly_params={"layout":{"title": {"text": SUBREDDIT}}})
    layout = go.Layout(
        xaxis=dict(range=range),
        yaxis=dict(range=range)
    )
    fig = fig.update(layout=layout)
    fig.show()
    return fig