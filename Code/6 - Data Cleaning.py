from creepy_module import clean_data, clean_comments

import os

PATH = '../Creepy Data/'
OUTPUT_PATH = '../pickles/'

for subreddit in os.listdir(PATH):
    # I have done nosleep, delete it in the future.
    if subreddit == 'NoSleep':
        continue
        
    try:
        os.makedirs(f"{OUTPUT_PATH + subreddit}")
    except FileExistsError:
        pass
    
    print(f'''
    -----------
    {subreddit}
    -----------
    ''')
    
    # Data
    subreddit_csv = clean_data(f"{PATH + subreddit}/{subreddit}/RS_2020_{subreddit.lower()}.csv")
    subreddit_csv.to_pickle(f"{OUTPUT_PATH + subreddit}/RS_2020_{subreddit.lower()}.pickle")
    
    # Comments
    subreddit_comments_csv = clean_comments(f"{PATH + subreddit}/{subreddit}/{subreddit.lower()}_comments.csv")
    subreddit_comments_csv.to_pickle(f"{OUTPUT_PATH + subreddit}/{subreddit.lower()}_comments.pickle")