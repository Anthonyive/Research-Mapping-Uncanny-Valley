import tensorflow as tf
import pandas as pd
import numpy as np
import spacy
import re
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
from tensorflow import keras
from ftfy import fix_text
import matplotlib
import swifter
from ftfy import fix_text
from langdetect import detect
from tqdm import tqdm
from tqdm.gui import tqdm as tqdm_gui
import swifter
from ftfy import fix_text
from langdetect import detect
from tqdm import tqdm
from tqdm.gui import tqdm as tqdm_gui
import swifter

def clean_data(csv_in, lang='en'):
    '''
    Cleans the reddit data in csv format and returns only one language specified.
    
    Input csv needs at least these columns:
        subreddit: str, subreddit name, usually repeated.
        id: str, post ids, usually are length six strings.
        created_utc: int, time created in UNIX time.
        title: str, post title.
        selftext: str, post body. 
        score: int, post upvotes.
        
    Main features:
        - Drop removed, deleted, and NAs
        - Remove links
        - Fix encoding problems and strip spaces
        - Filter out all languages except language specified

    Parameters
    ----------
    csv_in : str, required
        The path to the input csv.
    lang : str, optional
        Filter out the language specified. For example, if lang is 'en', then the cleaned data will only return posts that are detected as English. 

    Raises
    ------
    
    '''

    tqdm.pandas(ncols=50)  # can use tqdm_gui, optional kwargs, etc
    
    print('Reading data...')
    csv = pd.read_csv(csv_in, 
                      usecols = ['subreddit', 'id', 'created_utc', 'title', 'selftext', 'score'], 
                      dtype = {'subreddit': 'string', 
                               'id': 'string', 
                               'created_utc': 'object', 
                               'title': 'string', 
                               'selftext': 'string', 
                               'score': 'object'})
    csv.reset_index(inplace = True)
    csv.score = pd.to_numeric(csv.score,errors='coerce')
    print('Done')
    
    print('Removing nonexisting rows...')
    # drop removed, deleted, and NAs
    csv = csv[csv.selftext != '[removed]']
    csv = csv[csv.selftext != '[deleted]']
    csv.dropna(inplace=True)
    print('Done')

    '''
    This regex means:
    If there is a left paranthesis (or left quote) in front of http or https, it will match non whitespaces until the right parenthesis (or right quote).
    Otherwise, it will match http or https and non whitespaces until whitespaces.
    '''
    print('Removing links...')
    url_regex = r'''(\(|\")?(?:http|https)\S+(?(1)[\)|\"]|)'''
    csv.selftext = csv.selftext.str.replace(url_regex, '', flags=re.M)
    print('Done')
    
    '''
    # clean html, something like &amp;
    def clean_html(row):
        from bs4 import BeautifulSoup
        from html import unescape

        soup = BeautifulSoup(unescape(row), 'lxml')
        return soup.text
    
    csv.selftext = csv.selftext.progress_apply(clean_html)
    '''
    
    # above code is replaced by fix_text from module ftfy
    # example:
#     >>> print(fix_encoding("(à¸‡'âŒ£')à¸‡"))
#     (ง'⌣')ง
    csv.title = csv.title.swifter.progress_bar(enable=True, desc='Cleaning titles').apply(fix_text)
    csv.selftext = csv.selftext.swifter.progress_bar(enable=True, desc='Cleaning posts').apply(fix_text)
    
    # strip spaces
    csv.selftext = csv.selftext.str.replace('\u200b', ' ')
    csv.selftext = csv.selftext.str.strip()
    
    def detect_lang(row):
        try:
            return detect(row)
        except:
            return np.nan
    
    csv['title_language'] = csv.title.swifter.progress_bar(enable=True, desc='Detecting title language').apply(detect_lang)
    
    csv['selftext_language'] = csv.selftext.swifter.progress_bar(enable=True, desc='Detecting posts language').apply(detect_lang)
    
    # return posts that are written in english
    csv = csv.loc[csv.selftext_language == lang,:]
        
    return csv

# nosleep2020 = clean_data('./Creepy Data/NoSleep/NoSleep/RS_2020_nosleep.csv')


def clean_comments(csv_in):
    '''
    Cleans the reddit data's comments in csv format.
    
    Input csv needs at least these columns:
        parent_id: str, post id
        link_id: str, id of link
        body: str, comment body
        score: int, comment upvotes
        
    Main features:
        - Drop removed, deleted, and NAs
        - Drop indirect comments. Only direct comments will be kept
        - Remove links
        - Fix encoding problems and strip spaces
        - Filter out all languages except language specified

    Parameters
    ----------
    csv_in : str, required
        The path to the input csv.

    Raises
    ------
    
    '''
    
    tqdm.pandas(ncols=50)  # can use tqdm_gui, optional kwargs, etc
    
    print('Reading data...')
    csv = pd.read_csv(csv_in, 
                           usecols = ['parent_id', 'link_id', 'body', 'score'], 
                           dtype = {'parent_id': 'string', 'link_id': 'string', 'body': 'string', 'score': 'object'})
    csv.score = pd.to_numeric(csv.score,errors='coerce')
    print('Done')
    
    # drop removed, deleted, and NAs
    print('Removing nonexisting rows...')
    csv = csv[csv.body != '[removed]']
    csv = csv[csv.body != '[deleted]']
    csv.dropna(inplace=True)
    print('Done')
    
    # filter out indirect comments
    print('Filtering indirect comments...')
    direct_comment = csv.parent_id.str.startswith('t3')
    csv = csv.loc[direct_comment]
    print('Done')
    
    '''
    This regex means:
    If there is a left paranthesis (or left quote) in front of http or https, it will match non whitespaces until the right parenthesis (or right quote).
    Otherwise, it will match http or https and non whitespaces until whitespaces.
    '''
    print('Removing links...')
    url_regex = r'''(\(|\")?(?:http|https)\S+(?(1)[\)|\"]|)'''
    csv.body = csv.body.str.replace(url_regex, '', flags=re.M)
    print('Done')
    
    # example:
    # >>> print(fix_encoding("(à¸‡'âŒ£')à¸‡"))
    # (ง'⌣')ง
    csv.body = csv.body.swifter.progress_bar(enable=True, desc='Cleaning comments').apply(fix_text)    
    
    # strip spaces
    csv.body = csv.body.str.replace('\u200b', ' ')
    csv.body = csv.body.str.strip()
    
    def detect_lang(row):
        try:
            return detect(row)
        except:
            return np.nan
    
    csv['body_language'] = csv.body.swifter.progress_bar(enable=True, desc='Detecting comment language').apply(detect_lang)
    
    # return posts that are written in english
    csv = csv.loc[csv.body_language == 'en',:]
    
    return csv