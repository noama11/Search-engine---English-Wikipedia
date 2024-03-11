import math
import re
import builtins as pybuiltins
from contextlib import closing
from nltk.corpus import stopwords
from inverted_index_gcp import *
from nltk.stem.porter import *
from nltk.corpus import wordnet

import nltk
nltk.download('stopwords')

# stop words
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb', 'became', 'may']
RE_WORD = re.compile(r"""[\#\@\w\d](['\-]?\w){1,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)


def tokenize_stemming(text):
    list_token = []
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    stemmer = PorterStemmer()
    for token in tokens:
        if token not in english_stopwords:
            list_token.append(stemmer.stem(token))
    return list_token


def term_frequency(text, id):
    tokens = tokenize_stemming(text)
    word_freq = {}
    for token in tokens:
        if token not in all_stopwords:
            if token not in word_freq:
                word_freq[token] = 1
            else:
                word_freq[token] += 1
    lst_tuples = [(token, (id, freq)) for token, freq in word_freq.items()]
    return lst_tuples


# helper for normalized the score of anchor text
def min_max_normalize(dictionary):
    min_value = min(dictionary.values())
    max_value = max(dictionary.values())
    denominator = max_value - min_value
    if denominator == 0:
        return dictionary

    for key in dictionary:
        dictionary[key] = (dictionary[key] - min_value) / denominator

    return dictionary



def min_max_list(scores):
    # Extract the scores
    values = [score for _, score in scores]
    # print(values)
    # Get the maximum and minimum scores
    max_score = max(values)
    min_score = min(values)

    # Avoid division by zero
    if max_score == min_score:
        return [(doc_id, 0) for doc_id, _ in scores]

    # Normalize the scores
    normalized_scores = [
        (doc_id, (score - min_score) / (max_score - min_score))
        for doc_id, score in scores
    ]

    return normalized_scores


# weight = tf*idf
def calculate_tf_idf(index, term, tf):
    tf_idf = tf * (math.log2(index.N / index.df[term]))
    return tf_idf


def cosine_similarity(query, index):
    dict_cosine_sim = defaultdict(float)
    query_dict = dict(term_frequency(query, 0))
    query_list_keys = list(query_dict.keys())
    query_list_values = list(query_dict.values())
    index_df_list_keys = list(index.df.keys())

    for term in query_list_keys:
        w_term_query = (query_dict[term][1] / len(query)) * math.log2(index.N / index.df[term])

        if term in index_df_list_keys:
            posting_list = index.read_a_posting_list(".", term, "noam209263805")
            for doc_id, freq in posting_list:
                w_term_doc = calculate_tf_idf(index, term, freq / index.doc_len[doc_id])
                dict_cosine_sim[doc_id] += (w_term_doc) * (w_term_query)

    for doc_id in list(dict_cosine_sim.keys()):
        Word_doc_id__weight = index.norm[doc_id]
        dict_cosine_sim[doc_id] /= (
            math.sqrt(pybuiltins.sum(value[1] ** 2 for value in query_list_values) * Word_doc_id__weight))

    sorted_docs = sorted(dict_cosine_sim.items(), key=lambda x: x[1], reverse=True)
    top_100_docs = sorted_docs[:100]
    top_100_norm = min_max_list(top_100_docs)

    return top_100_norm


def bm25_score(query, index):
    doc_BM25_value = defaultdict(float)
    query = tokenize_stemming(query)
    K = 1.5
    B = 0.75
    AVGDL = 341.0890174848911

    for token in query:
        token_df = index.df[token]
        token_idf = math.log2(index.N / token_df)

        posting_list = index.read_a_posting_list(".", token, "noam209263805")
        for page_id, word_freq in posting_list:
            numerator = word_freq * (K + 1)
            denominator = word_freq + K * (1 - B + (B * index.doc_len[page_id]) / AVGDL)
            doc_BM25_value[page_id] += token_idf * (numerator / denominator)

    sorted_docs = sorted(doc_BM25_value.items(), key=lambda x: x[1], reverse=True)
    top_100_docs = sorted_docs[:100]
    top_100_norm = min_max_list(top_100_docs)

    return top_100_norm


def generate_word_2grams(words):
    return {' '.join(words[i:i + 2]) for i in range(len(words) - 1)}


# if ngram == true -> do title_2_words
# if ngram == false -> do title_1
def search_title(query, index, index_2, ngram=False):
    dict_cosine_sim = defaultdict(float)
    query_filtered_list = tokenize_stemming(query)

    if not ngram:
        for term in query_filtered_list:
            if term in index.df:
                posting_list = index.read_a_posting_list(".", term, "noam209263805")
                for doc_id, freq in posting_list:
                    x = re.sub(r'[^\w]', ' ', index.doc_id_title[doc_id]).split(" ")
                    dict_cosine_sim[doc_id] += freq / len(x)
    else:
        query_2gram_set = generate_word_2grams(query_filtered_list)

        for two_word_query in query_2gram_set:
            if two_word_query in index_2.df:
                posting_list = index_2.read_a_posting_list(".", two_word_query, "noam209263805")
                for doc_id, freq in posting_list:
                    words = tokenize_stemming(index.doc_id_title[doc_id])
                    title_2gram_set = generate_word_2grams(words)
                    jccard = freq / len(query_2gram_set.union(title_2gram_set))
                    dict_cosine_sim[doc_id] += jccard
    if not dict_cosine_sim:
      return []
    sorted_docs = sorted(dict_cosine_sim.items(), key=lambda x: x[1], reverse=True)
    top_100_docs = sorted_docs[:100]
    top_100_norm = min_max_list(top_100_docs)

    return top_100_norm


def search_anchor(query, index):
    dict_cosine_sim = defaultdict(float)
    query_list_clean = tokenize_stemming(query)
    for term in query_list_clean:
        if term in index.df:
            posting_list = index.read_a_posting_list(".", term, "noam209263805")
            for doc_id, freq in posting_list:
                dict_cosine_sim[doc_id] += freq

    if not dict_cosine_sim:
      return []
    sorted_docs = sorted(dict_cosine_sim.items(), key=lambda x: x[1], reverse=True)
    top_100_docs = sorted_docs[:100]
    top_100_norm = min_max_list(top_100_docs)
    return top_100_norm




def min_max_body_2(scores):
    # Extract the values to be normalized
    values = [score[0] for _, score in scores]

    # Get the maximum and minimum values
    max_score = max(values)
    min_score = min(values)

    # Avoid division by zero
    if max_score == min_score:
        return [(doc_id, (0, tf)) for doc_id, (_, tf) in scores]

    # Normalize the values
    normalized_scores = [
        (doc_id, ((value - min_score) / (max_score - min_score), tf))
        for doc_id, (value, tf) in scores
    ]

    return normalized_scores


def search_body_2(query, index_body1, index_body2):
    dict_of_tuples = defaultdict(lambda: (0, 0))
    query_filtered_list = tokenize_stemming(query)
    query_2gram_set = generate_word_2grams(query_filtered_list)

    for two_word_query in query_2gram_set:
        if two_word_query in index_body2.df:
            posting_list = index_body2.read_a_posting_list(".", two_word_query, "noam209263805")
            for doc_id, freq in posting_list:
                jccard = freq / index_body1.doc_len[doc_id]

                current_tuple = dict_of_tuples[doc_id]  # Retrieve the current tuple
                new_tuple = (current_tuple[0] + jccard, current_tuple[1] + 1)  # Modify the tuple
                dict_of_tuples[doc_id] = new_tuple  # Update the dictionary

    if not dict_of_tuples:
      return []
    # Sorting the top 200 tuples according to sequence count
    sorted_docs = sorted(dict_of_tuples.items(), key=lambda x: x[1][1], reverse=True)
    top_200_docs = sorted_docs[:200]

    # Sorting the top 100 tuples according to TF
    top_100_tf_sorted = sorted(top_200_docs, key=lambda x: x[1][0], reverse=True)[:100]

    top_100_norm = min_max_body_2(top_100_tf_sorted)
    return top_100_norm


def search_res(inverted_title, inverted_title_2_words, inverted_body, inverted_body_2_words, inverted_anchor, query):
    query_filtered = tokenize_stemming(query)
    res_dict = defaultdict(float)

    if len(query_filtered) == 1:
        res_dict = score_one_word(inverted_title, inverted_body, inverted_title_2_words, inverted_anchor, query)

    if len(query_filtered) > 1:
        score_dic_res = score2(inverted_title, inverted_body, inverted_title_2_words, inverted_anchor, query)
        score_title_2_ngrams = search_title(query, inverted_title, inverted_title_2_words, ngram=True)
        score_body_2_ngrams = search_body_2(query, inverted_body, inverted_body_2_words)

        for doc_id, score in score_dic_res.items():
            res_dict[doc_id] += score * 1/10

        for doc_id, score in score_title_2_ngrams:
            res_dict[doc_id] += score * 9/10

        for doc_id, score in score_body_2_ngrams:
            res_dict[doc_id] += 2/10


    sorted_docs = sorted(res_dict.items(), key=lambda x: x[1], reverse=True)
    top_100_docs = sorted_docs[:60]

    # Return a list of tuples containing the document ID and its title
    result = [(str(doc_id), inverted_title.doc_id_title[doc_id]) for doc_id, _ in top_100_docs]
    return result


# hepler functions for search_res
# query len = 1
def score_one_word(inverted_title, inverted_body, inverted_title_2_words, inverted_anchor, query):
    score_title = search_title(query, inverted_title, inverted_title_2_words, ngram=False)
    score_anchor = search_anchor(query, inverted_anchor)

    res_dict = defaultdict(float)
    for doc_id, score in score_title:
        res_dict[doc_id] += score * 12/20

    for doc_id, score in score_anchor:
        res_dict[doc_id] += score * 8/20

    return res_dict


# n gram = false -> title1
def score2(inverted_title, inverted_body, inverted_title_2_words, inverted_anchor, query):
    score_title = search_title(query, inverted_title, inverted_title_2_words, ngram=False)
    score_body = bm25_score(query, inverted_body)
    score_anchor = search_anchor(query, inverted_anchor)
    res_dict = defaultdict(float)

    for doc_id, score in score_title:
        res_dict[doc_id] += score * 7 / 16

    for doc_id, score in score_body:
        res_dict[doc_id] += score * 2 / 16

    for doc_id, score in score_anchor:
        res_dict[doc_id] += score * 7 / 16

    return res_dict
