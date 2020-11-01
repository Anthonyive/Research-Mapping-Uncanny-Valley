import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import pandas as pd
import plotly.graph_objs as go
import numpy as np

pd.options.plotting.backend = "plotly"

creepy = pd.read_pickle('./pickles/creepy.pickle')
noncreepy = pd.read_pickle('./pickles/noncreepy.pickle')

creepy_sum_vec_with_log_prepended = creepy.loc[:, 'vec'].copy()
noncreepy_sum_vec_with_log_prepended = noncreepy.loc[:, 'vec'].copy()

creepy_features = pd.DataFrame(
    creepy_sum_vec_with_log_prepended.to_list()).to_numpy(dtype=float)
creepy_labels = np.ones(len(creepy_features))

noncreepy_features = pd.DataFrame(
    noncreepy_sum_vec_with_log_prepended.to_list()).to_numpy(dtype=float)
noncreepy_labels = np.zeros(len(noncreepy_features))

creepy_features_df = pd.DataFrame(creepy_features).head(10).transpose()
fig = creepy_features_df.plot(title="Visualization of creepy vectors",
                              labels=dict(index="vector dimensions", value="value", variable="story"))
fig.show()

fig.show()
