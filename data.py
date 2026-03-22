import pandas as pd
import re
import string
import nltk
import networkx as nx
from nltk.corpus import stopwords
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.update(['and', 'amp', 'rt', 'th', 'nt', 'via']) 
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# --------------------------------------------------------------------- #
def clean_tweet(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Process with spaCy
    doc = nlp(text)
    # Extract lemmas, filter stopwords, punctuation, and short words
    cleaned_tokens = []
    for token in doc:
        # Get the lemma (root form)
        lemma = token.lemma_     
        # 1. Check Custom Mapping first
        #lemma = custom_map.get(lemma, lemma)
        
        # 2. Filtering
        if (not token.is_stop and 
            not token.is_punct and 
            len(lemma) > 2 and 
            lemma.isalpha()):
            cleaned_tokens.append(lemma)            
    
    return " ".join(cleaned_tokens)
# --------------------------------------------------------------------- #

def vectorizer(stopwords, df_list):
    tfidf = TfidfVectorizer(max_features=2000, stop_words=list(stopwords))
    tfidf_matrix = tfidf.fit_transform(df_list)
    word_scores = tfidf_matrix.sum(axis=0).A1
    words = tfidf.get_feature_names_out()

    # Get the list of the top 200 words
    top_200_df = pd.DataFrame({'word': words, 'score': word_scores}).head(200)
    #top_200_df.to_excel('IranIsrael/top_200_words.xlsx', index=False)
    top_200_list = top_200_df.sort_values(by='score', ascending=False)
    
    return top_200_list
# --------------------------------------------------------------------- #

def create_net(vocab, data):
    if isinstance(vocab, pd.DataFrame):
        vocab_list = vocab.iloc[:, 0].astype(str).tolist() # Takes first column
    elif isinstance(vocab, pd.Series):
        vocab_list = vocab.astype(str).tolist()
    else:
        vocab_list = list(vocab)
    
    cv = CountVectorizer(vocabulary=vocab_list, binary=True)
    word_doc_matrix = cv.fit_transform(data)
    co_occurrence_matrix = (word_doc_matrix.T @ word_doc_matrix)
    co_occurrence_matrix.setdiag(0)  

    G = nx.Graph()
    co_occ = co_occurrence_matrix.tocoo()

    for i, j, weight in zip(co_occ.row, co_occ.col, co_occ.data):
        if weight > 2 and i < j :
            G.add_edge(vocab_list[i], vocab_list[j], weight=weight)

    edge_list = []
    for u, v, d in G.edges(data=True):
        edge_list.append({'Source': u, 'Target': v, 'Weight': d['weight']})

    edge_df = pd.DataFrame(edge_list)
    
    return G, edge_df

# --------------------------------------------------------------------- #

