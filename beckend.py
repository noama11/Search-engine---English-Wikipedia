import math
import re
from contextlib import closing

from nltk.corpus import stopwords
from inverted_index_colab import *

from inverted_index_colab import MultiFileReader


def read_posting_list(inverted, bucket_name, w):
    """
    Read posting list of word from bucket store

    :param  inverted: inverted index object
            name_bucket: name bucket that store the inverted index
            w: the requested word

    :return posting list - list of tuple (doc_id,tf)
    """
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        try:
            b = reader.read(locs, inverted.df[w] * 6)
        except: return []
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * 6:i * 6 + 4], 'big')
            tf = int.from_bytes(b[i * 6 + 4:(i + 1) * 6], 'big')
            posting_list.append((doc_id, tf))
        return posting_list

def term_frequency(text, id):
    ''' Count the frequency of each word in `text` (tf) that is not included in
    `all_stopwords` and return entries that will go into our posting lists.
    Parameters:
    -----------
        text: str
            Text of one document
        id: int
            Document id
    Returns:
    --------
        List of tuples
            A list of (token, (doc_id, normalized_tf)) pairs
            for example: [("Anarchism", (12, 0.5)), ...]
    '''

    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']
    all_stopwords = english_stopwords.union(corpus_stopwords)

    # Count the frequency of each word that is not a stopword
    tokens = text
    word_freq = {}
    for token in tokens:
        if token not in all_stopwords:
            if token not in word_freq:
                word_freq[token] = 1
            else:
                word_freq[token] += 1
    lst_tuples = [(token, (id, freq)) for token, freq in word_freq.items()]

    return lst_tuples


# weight = tf*idf
def calculate_tf_idf(index, term, tf, doc_id):
  tf_idf = (tf/index.doc_len[doc_id])* math.log2(index.N/index.df[term])
  return tf_idf

def cosine_similarity(query, index):
    """ Returns: {doc_id:cosine score} """
    dict_cosine_sim = {}
    doc_weights_dict = {}
    query_dict = dict(term_frequency(query ,0))
    for term in query_dict.keys():
      w_term_query = (query_dict[term][1]/len(query))* math.log2(index.N/index.df[term])
      if term in index.df.keys():
        posting_lst = read_posting_list(index,term)
        for doc_id, freq in posting_lst:
          w_term_doc = calculate_tf_idf(index, term, freq, doc_id)
          if doc_id in dict_cosine_sim.keys():
            dict_cosine_sim[doc_id] += (w_term_doc)*(w_term_query)
            doc_weights_dict[doc_id] += (w_term_doc)*(w_term_doc)
          else:
            dict_cosine_sim[doc_id] = (w_term_doc)*(w_term_query)
            doc_weights_dict[doc_id] = (w_term_doc)*(w_term_doc)

    for doc_id in dict_cosine_sim.keys():
        dict_cosine_sim[doc_id] *=  math.sqrt(sum(score for score in doc_weights_dict.value()) * sum([math.pow(score, 2) for score in query_dict.values()]))

    sorted_docs = sorted(dict_cosine_sim.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs
