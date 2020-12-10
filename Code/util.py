from numba import cuda
import spacy
nlp = spacy.load('en_core_web_lg')
from sentence_transformers import SentenceTransformer, util
import numpy as np

def release_vram():
	# clear memory of gpu
	device = cuda.get_current_device()
	device.reset()
    
def semantic_search(document, model='distilbert-base-nli-stsb-mean-tokens', verbose=False, sort_by='high_to_low', first_n = None):
    doc = nlp(document)
    embedder = SentenceTransformer(model)
    
    # https://www.pnas.org/content/pnas/114/38/E7900.full.pdf
    # 27 emotions
    # Corpus with example sentences
    emotions = ['Admiration',
                'Adoration',
                'Aesthetic Appreciation',
                'Amusement',
                'Anxiety',
                'Awe',
                'Awkwardness',
                'Boredom',
                'Calmness',
                'Confusion',
                'Craving',
                'Disgust',
                'Empathetic pain',
                'Entrancement',
                'Envy',
                'Excitement',
                'Fear',
                'Horror',
                'Interest',
                'Joy',
                'Nostalgia',
                'Romance',
                'Sadness',
                'Satisfaction',
                'Sexual desire',
                'Sympathy',
                'Triumph',
              ]
    emotions_embeddings = embedder.encode(emotions, convert_to_tensor=True)

    # sentences in a document
    sentences = [repr(sent) for sent in doc.sents]
    # sentences = ['A man held a knife and cut his fingers.', 
    #              'A woman held a knife and cut her fingers.', 
    #             'The body is bleeding.']

    sents_emotion_vec = []

    for idx,sentence in enumerate(sentences):
        sentence_embedding = embedder.encode(sentence, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(sentence_embedding, emotions_embeddings)[0]
        cos_scores = cos_scores.cpu()

        #We use torch.topk to find the highest 5 scores
        #top_results = torch.topk(cos_scores, k=top_k)
        
        if verbose:
            print("\n\n======================\n")
            print("Query:", sentence)
            print("\nSimilarities between query sentence and 27 emotions:")

        sent_emotion_vec = []
        
        if sort_by == 'high_to_low':
            # sort emotions based on similarity:
            for emotion, score in sorted(zip(emotions, cos_scores), key = lambda t: t[1], reverse= True):
                if verbose:
                    print(emotion.strip(), "(Score: %.4f)" % (score))
                sent_emotion_vec.append(score.numpy().item())
        elif sort_by == 'low_to_high':
            # sort emotions based on similarity:
            for emotion, score in sorted(zip(emotions, cos_scores), key = lambda t: t[1], reverse= False):
                if verbose:
                    print(emotion.strip(), "(Score: %.4f)" % (score))
                sent_emotion_vec.append(score.numpy().item())
        elif sort_by == 'alphabetical':
            # sort emotions alphabetically:
            for emotion, score in zip(emotions, cos_scores):
                if verbose:
                    print(emotion.strip(), "(Score: %.4f)" % (score))
                sent_emotion_vec.append(score.numpy().item())
        else:
            print("sort_by only has three candidates: 'high_to_low', 'low_to_high', and 'alphabetical'")
            return None

        sents_emotion_vec.append(sent_emotion_vec)
        
        if first_n is not None and idx>first_n-2:
            break
        
    return np.array(sents_emotion_vec)