import sys
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from time import time
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import time
import math
from functools import reduce
# from google.cloud import storage
from inverted_index_gcp import *
import hashlib


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


# Put your bucket name below and make sure you can access it without an error
bucket_name = '207219783saturday'


# stop words
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb', 'became', 'may']
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)


# tokenize and stemming
def tokenize_stemming(text):
    list_token = []
    stemmer = PorterStemmer()
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    for token in tokens:
        if token not in english_stopwords:
            list_token.append(stemmer.stem(token))
    return list_token


# tf
def term_frequency(text, id):
    tokens = tokenize_stemming(text)

    # Count the frequency of each word that is not a stopword
    word_freq = {}
    doc_size = 0

    for token in tokens:
        doc_size += 1
        if token not in all_stopwords:
            if token not in word_freq:
                word_freq[token] = 1
            else:
                word_freq[token] += 1

    lst_tuples = [(token, (id, freq)) for token, freq in word_freq.items()]
    return lst_tuples

#
# def reduce_word_counts(unsorted_pl):
#   sorted_pl = sorted(unsorted_pl, key=lambda x: x[0])
#   return sorted_pl
#
# # df
# def calculate_df(postings):
#   rdd_token_doc_freq = postings.map(lambda x: (x[0], len(x[1])))
#   return rdd_token_doc_freq
#
#
# # doc len
# def doc_lenght(text, doc_id):
#   tokens = tokenize_stemming(text)
#   return((doc_id,len(tokens)))

# NUM_BUCKETS = 124
# def token2bucket_id(token):
#   return int(_hash(token),16) % NUM_BUCKETS
#
# def partition_postings_and_write(postings, index_name):
#     bucketed_postings = postings.map(lambda x: (token2bucket_id(x[0]), [(x[0], x[1])]))
#     grouped_postings = bucketed_postings.reduceByKey(lambda x, y: x + y)
#     posting_locations = grouped_postings.map(lambda x: InvertedIndex.write_a_posting_list(x, f'bucket_{index_name}', bucket_name))
#     return posting_locations


# weight = tf*idf
def calculate_tf_idf(index, term, tf):
  tf_idf = tf* (math.log2(index.N/index.df[term]))
  return tf_idf


def cosine_similarity(query, index):
    """ Returns: {doc_id:cosine score} """

    dict_cosine_sim = {}
    doc_weights_dict = {}
    query_dict = dict(term_frequency(query, 0))

    # create lists of keys and values from the query_dict
    query_list_keys = list(query_dict.keys())
    query_list_values = list(query_dict.values())

    # create list of items from index.df dictionery keys
    index_df_list_keys = list(index.df.keys())

    for term in query_list_keys:
      w_term_query = (query_dict[term][1]/len(query))* math.log2(index.N/index.df[term])

      if term in index_df_list_keys:
        posting_list = index.read_a_posting_list(".", term, bucket_name)

        for doc_id, freq in posting_list:
          w_term_doc = calculate_tf_idf(index, term, freq/index.doc_len[doc_id])
          if doc_id in dict_cosine_sim.keys():
            dict_cosine_sim[doc_id] += (w_term_doc)*(w_term_query)
            doc_weights_dict[doc_id] += (w_term_doc) ** 2
          else:
            dict_cosine_sim[doc_id] = (w_term_doc)*(w_term_query)
            doc_weights_dict[doc_id] = (w_term_doc) ** 2

    for doc_id in list(dict_cosine_sim.keys()):
      Word_doc_id__weight = doc_weights_dict[doc_id]
      dict_cosine_sim[doc_id] /= (math.sqrt(sum(value[1] ** 2 for value in query_list_values) * Word_doc_id__weight ))

    sorted_docs = sorted(dict_cosine_sim.items(), key=lambda x: x[1], reverse=True)
    top_100_docs = sorted_docs[:100]

    return top_100_docs

# def BM25(query, index, k, b, avg_doclen):
#     bm_score = {}
#     # create lists of keys and values from the query_dict
#     query_list_keys = list(query_dict.keys())
#     query_list_values = list(query_dict.values())
#     # create list of items from index.df dictionery keys
#     index_df_list_keys = list(index.df.keys())

#     idf_dict = calc_idf(query, N, index)

#     for term in query_list_keys:
#         if term in index_df_list_keys:
#             posting_list = index.read_a_posting_list(".", term, bucket_name)
#             for doc_id, freq in posting_list:
#                 B = 1 - b + b * (index.dl[doc_id] / avg_doclen)
#                 idf = idf_dict[term]
#                 tf = freq / index.dl[doc_id]
#                 if doc_id in bm_score:
#                     bm_score[doc_id] += (idf * freq * (k + 1)) / ((freq + B * k))
#                 else:
#                     bm_score[doc_id] = (idf * freq * (k + 1)) / ((freq + B * k))

#     return dict(sorted(bm_score.items(), key=lambda x: x[1], reverse=True))


def search(inverted, query):
    """ Returns: n best docs """
    start_time = time.time()  # Record the start time
#     query_filtered = tokenize_stemming(query)
    cosSim_score = cosine_similarity(query, inverted)
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time
    print("Function execution time:", execution_time, "seconds")
    return cosSim_score

############### test ##################
# create_index("title", doc_title_pairs)
# create_index("body", doc_body_pairs)
#
# idx_title = InvertedIndex.read_index("bucket_title", 'index_title', bucket_name)
# idx_body = InvertedIndex.read_index("bucket_body", 'index_body', bucket_name)

# query = "genetics"
# # query = "Who is considered the"
# # query = "economic"
# res_t = search(idx_title, query)
# res_b = search(idx_body, query)

