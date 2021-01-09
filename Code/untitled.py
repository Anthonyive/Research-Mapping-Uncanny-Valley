import pandas as pd
from sentence_transformers import SentenceTransformer
import nltk
import swifter
from tqdm.notebook import tqdm
tqdm.pandas()
import pickle

def sbert_sum(row):
    return sum([sbert.encode(sent) for sent in nltk.sent_tokenize(row['selftext'])])

file=['NoSleep','Confession']

for name in file:
    df = pd.read_csv(f'Download/Cleaned Data/{name}_cleaned.csv')
    df = df.loc[df['created_utc']>1546300800].copy()

    sbert = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    df['sent_count'] = df.swifter.progress_bar(enable=True, desc='Counting sentences').apply(lambda row: len(nltk.sent_tokenize(row['selftext'])), axis=1)
    df['sbert_sum'] = df.progress_apply(sbert_sum, axis=1)
    df['sbert_avg'] = df.swifter.progress_bar(enable=True, desc='Calculating averge').apply(lambda row: row['sbert_sum']/row['sent_count'], axis=1)
    
    pickle.dump(df, open(f'Download/Cleaned Data/{name}.pickle', 'wb'))