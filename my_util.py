import sentence_transformers
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import spacy
from tqdm import tqdm, tqdm_pandas, tqdm_notebook
tqdm.pandas()

nlp = spacy.load('en_core_web_lg')
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

def clean_data(csv_in):
    csv_read = pd.read_csv(csv_in)

    csv = csv_read[['id', 'title', 'selftext', 'score']]
    csv = csv[csv.selftext != '[removed]']
    csv = csv[csv.selftext != '[deleted]']
    csv.dropna(subset = ["selftext"], inplace=True)
    csv = csv.replace(r'^\s+','', regex=True)
    return csv

def clean_comments(csv_in):
    csv_read = pd.read_csv(csv_in, dtype = str)
    csv = csv_read[csv_read['score'].apply(pd.to_numeric, errors='coerce').notna()]
    csv = csv[['parent_id', 'body', 'score']]
    csv = csv[csv.body != '[removed]']
    csv = csv[csv.body != '[deleted]']
    csv.dropna(subset = ["body"], inplace=True)
    csv = csv.replace(r'^\s+','', regex=True)
    
    # t3_ are direct comments
    csv = csv[csv.parent_id.str.startswith('t3_')]
    parent_id = csv.parent_id.str.split(r"_", expand=True)
    
    result = pd.concat([csv, parent_id], axis=1)
    result = result[[1, 'body', 'score']]
    result.columns = ['parent_id', 'body', 'score']
    return result

# using sBert model and spacy to get sum of sentence vectors.
def sum_vec(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    comment_vec = model.encode(sentences)
    return np.sum(comment_vec, axis = 0)

# find the linear combination of vectors of comments
# for example, (sum vec of a comment) * score, and add them all up for each post
def lc_comments(comments_df):
    comments_df.loc[:,'sum_vec'] = comments_df.progress_apply(lambda x: sum_vec(x['body']), axis=1)

    comments_df.loc[:,'score'] = pd.to_numeric(comments_df.loc[:,'score'])

    # print(comments_df_short.score.dtypes)
    comments_df['lc'] = comments_df['score'].to_numpy() * comments_df['sum_vec'].to_numpy()
    comments_df.groupby(['parent_id'])['lc'].apply(np.sum)
    
    return comments_df

# sum vec for the post data frame
def sum_vec_df(df):
    df.loc[:,'sum_vec'] = df.progress_apply(lambda x: sum_vec(x['selftext']), axis=1)
    return df

def lc(df, comments_df):
    df_merged = lc_comments(comments_df).merge(sum_vec_df(df), left_on='parent_id', right_on='id')
    df_merged.columns = ['parent_id', 'comment', 'c_score', 'c_sum_vec', 'c_lc', 'id', 'title', 'selftext', 't_score', 't_sum_vec']
    comment_lc = pd.DataFrame(df_merged.groupby(['id'])['c_lc'].apply(np.sum))
    # comment_lc.reset_index(inplace=True) # Resets the index, makes factor a column
    # comment_lc.drop("comment",axis=1,inplace=True) # drop factor from axis 1 and make changes permanent by inplace=True
    
    df_merged_gb_id = df_merged.groupby(['id']).first()
    df_merged_gb_id['t_lc'] = df_merged_gb_id['t_score'].to_numpy() * df_merged_gb_id['t_sum_vec'].to_numpy()
    df_merged_gb_id.groupby(['parent_id'])['t_lc'].apply(np.sum)
    
    text_lc = df_merged_gb_id.loc[:,['t_lc']]
    
    lc = pd.merge(text_lc, comment_lc, on='id', how = 'outer')
    lc['output'] = lc.apply(np.sum, axis = 1)
    return lc