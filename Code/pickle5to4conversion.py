import os
import pandas as pd

PATH = "../pickles/"

# data_cleaned is a dictionary of pandas data frames
data_cleaned = dict()
for subreddit in os.listdir(PATH):
    if subreddit.startswith('.'):
        continue
    data_cleaned[f"RS_2020_{subreddit.lower()}"] = pd.read_pickle(f"{PATH + subreddit}/RS_2020_{subreddit.lower()}.pickle")
    data_cleaned[f"{subreddit.lower()}_comments"] = pd.read_pickle(f"{PATH + subreddit}/{subreddit.lower()}_comments.pickle")


import pickle

with open('../pickles/data_cleaned.pickle', 'wb') as handle:
    pickle.dump(data_cleaned, handle, protocol=4)