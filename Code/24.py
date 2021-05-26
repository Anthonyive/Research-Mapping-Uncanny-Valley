import pandas as pd
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

df = pd.read_csv('Code/Downloads/MFTD.csv')
data = df['content'].to_list()

embeddings = model.encode(data, show_progress_bar=True)