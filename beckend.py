import math
import re
import builtins as pybuiltins
from contextlib import closing

import nltk
from nltk.corpus import stopwords
from inverted_index_gcp import *
from nltk.stem.porter import *
from nltk.corpus import wordnet
nltk.download('wordnet')



def read_posting_list(inverted, bucket_name, w):
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

# stop words
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb', 'became', 'may']
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)

######## new #######
def query_expansion(query):
    synonyms = {}
    hyponyms = {}

    for word in query:
        word_synsets = wordnet.synsets(word)
        for synset in word_synsets:
            syn = synset.lemmas()[0].name().lower()
            if syn and syn != word:
                synonyms[word] = syn
                break

    for word in query:
        word_synsets = wordnet.synsets(word)
        for synset in word_synsets:
            try:
                hyp = synset.hyponyms()[0].name()
                word_synset = wordnet.synset(hyp)
                hypernyms = word_synset.hypernyms()
                lst = ([synset.name().split(".")[0] for synset in hypernyms])
                h = lst[0].lower()
                if h != word:
                    hyponyms[word] = h
            except:
                pass
    for _, val in synonyms.items():
        if val not in query:
            query.append(val)
    for _, val in hyponyms.items():
        if val not in query:
            query.append(val)

    return query


######## new #######

# tokenize and stemming
def tokenize_stemming(text,query_exp=False):
    list_token = []
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]

    ######
    if(query_exp):
        tokens = query_expansion(tokens)
    ######

    stemmer = PorterStemmer()
    for token in tokens:
        if token not in english_stopwords:
            list_token.append(stemmer.stem(token))
    return list_token


def term_frequency(text, id, query_expansion = False):
    tokens = tokenize_stemming(text, query_expansion)
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
def calculate_tf_idf(index, term, tf):
  tf_idf = tf* (math.log2(index.N/index.df[term]))
  return tf_idf


def cosine_similarity(query, index):
    """ Returns: {doc_id:cosine score} """

    dict_cosine_sim = defaultdict(float)
    # doc_weights_dict = defaultdict(float)
    query_dict = dict(term_frequency(query, 0))

    # create lists of keys and values from the query_dict
    query_list_keys = list(query_dict.keys())
    query_list_values = list(query_dict.values())

    # create list of items from index.df dictionery keys
    index_df_list_keys = list(index.df.keys())

    for term in query_list_keys:
      w_term_query = (query_dict[term][1]/len(query))* math.log2(index.N/index.df[term])

      if term in index_df_list_keys:
        posting_list = index.read_a_posting_list(".", term, "noam209263805")
        
        for doc_id, freq in posting_list:  
          w_term_doc = calculate_tf_idf(index, term, freq/index.doc_len[doc_id])
          dict_cosine_sim[doc_id] += (w_term_doc)*(w_term_query)

    if (index == "body"):
        for doc_id in list(dict_cosine_sim.keys()):
            Word_doc_id__weight = index.norm[doc_id]
            dict_cosine_sim[doc_id] /= (
                math.sqrt(pybuiltins.sum(value[1] ** 2 for value in query_list_values) * Word_doc_id__weight))

    sorted_docs = sorted(dict_cosine_sim.items(), key=lambda x: x[1], reverse=True)
    top_100_docs = sorted_docs[:100]

    return top_100_docs

################


# def BM25(query, index, k, b, avg_doclen):
#     bm_score = {}
#     # create lists of keys and values from the query_dict
#     query_list_keys = list(query_dict.keys())
#     query_list_values = list(query_dict.values())
#     # create list of items from index.df dictionery keys
#     index_df_list_keys = list(index.df.keys())
#
#     idf_dict = calc_idf(query, N, index)
#
#     for term in query_list_keys:
#         if term in index_df_list_keys:
#             posting_list = index.read_a_posting_list(".", term, "noam209263805")
#             for doc_id, freq in posting_list:
#                 B = 1 - b + b * (index.dl[doc_id] / avg_doclen)
#                 idf = idf_dict[term]
#                 tf = freq / index.dl[doc_id]
#                 if doc_id in bm_score:
#                     bm_score[doc_id] += (idf * freq * (k + 1)) / ((freq + B * k))
#                 else:
#                     bm_score[doc_id] = (idf * freq * (k + 1)) / ((freq + B * k))
#
#     return dict(sorted(bm_score.items(), key=lambda x: x[1], reverse=True))

###############3

def search_title(inverted):
    res = []
    return res

def search_anchor(inverted):
    res = []
    return res


def search_res(inverted_title, inverted_body, inverted_anchor, query):
    score_title = cosine_similarity(query, inverted_title)
    score_body = cosine_similarity(query, inverted_body)
    # score_anchor = cosine_similarity(query, inverted_anchor)

    res_dict = defaultdict(float)
    
    for doc_id, score in score_title:
        res_dict[doc_id] += score * 7/16
    
    for doc_id, score in score_body:
        res_dict[doc_id] += score * 1/16

    # for doc_id, score in score_anchor:
    #     res_dict[doc_id] += 7/16


    sorted_docs = sorted(res_dict.items(), key=lambda x: x[1], reverse=True)

    # Select the top 100 documents
    top_100_docs = sorted_docs[:100]

    # Return a list of tuples containing the document ID and its title
    result = [(str(doc_id), inverted_title.doc_id_title[doc_id]) for doc_id, _ in top_100_docs]
    return result

