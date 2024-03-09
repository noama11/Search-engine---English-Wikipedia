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


##### normalizion ####
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
#####################

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
    return top_100_docs


def bm25_score(query, index):
    bm_score = defaultdict(float)
    score = 0.0
    k1 = 1.5
    b = 0.75
    avg_doc_len = 341.0890174848911  ## need to add a function!
    for term in query:
        posting_list = index.read_a_posting_list(".", term, "noam209263805")
        for doc_id, freq in posting_list:
            doc_len = index.doc_len[doc_id]
            idf = math.log2(index.N / index.df[term])
            numerator = idf * freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * doc_len / avg_doc_len)
            score += (numerator / denominator)
            bm_score[doc_id] += score

    # Normalize the values in dict_cosine_sim
    dict_bm25_score_normalized = min_max_normalize(bm_score)

    sorted_docs = sorted(dict_bm25_score_normalized.items(), key=lambda x: x[1], reverse=True)
    top_100_docs = sorted_docs[:100]
    return top_100_docs


def generate_word_2grams(words):
    return {' '.join(words[i:i+2]) for i in range(len(words) - 1)}
 

# if ngram == true -> do title_2_words
# if ngram == false -> do title_1
def search_title(query, index, index_2, ngram=False):
    dict_cosine_sim = defaultdict(float)
    # query_dict = dict(term_frequency(query, 0))
    query_filtered_list = tokenize_stemming(query)
    # query_list_keys = list(query_dict.keys())
    # index_df_list_keys = list(index.df.keys())

    if not ngram:
        # index_df_list_keys = list(index.df.keys())
        for term in query_filtered_list:
            if term in index.df:
                posting_list = index.read_a_posting_list(".", term, "noam209263805")
                for doc_id, freq in posting_list:
                    x = re.sub(r'[^\w]', ' ', index.doc_id_title[doc_id]).split(" ")
                    dict_cosine_sim[doc_id] += freq / len(x)
    else:
        # index_df_list_keys = list(index_2.df.keys())

        query_2gram_set = generate_word_2grams(query_filtered_list)

        for two_word_query in query_2gram_set:
            if two_word_query in index_2.df:
                posting_list = index_2.read_a_posting_list(".", two_word_query, "noam209263805")
                for doc_id, freq in posting_list:
                    # x = re.sub(r'[^\w]', ' ', index.doc_id_title[doc_id]).split(" ")
                    
                    words = tokenize_stemming(index.doc_id_title[doc_id])
                    title_2gram_set = generate_word_2grams(words)
                    
                    jccard = freq / len(query_2gram_set.union(title_2gram_set))
                    
                    # print("doc id: " + str(doc_id) + ", title: " + str(words) + ", jaccard = " + str(jccard) + " / " + str(len(query_2gram_set.union(title_2gram_set))))
                    dict_cosine_sim[doc_id] += jccard
                    
    sorted_docs = sorted(dict_cosine_sim.items(), key=lambda x: x[1], reverse=True)
    top_100_docs = sorted_docs[:100]
    return top_100_docs
    
def search_anchor(query, index):
    dict_cosine_sim = defaultdict(float)
    # query_dict = dict(term_frequency(query, 0))
    # query_terms_set = set(query_dict.keys())

    # index_df_set = set(index.df.keys())
    query_list_clean = tokenize_stemming(query)
    # Only process terms that exist in both the query and index
    # common_terms = query_terms_set.intersection(index_df_set)
    for term in query_list_clean:
        if term in index.df:
            posting_list = index.read_a_posting_list(".", term, "noam209263805")
            for doc_id, freq in posting_list:
                dict_cosine_sim[doc_id] += freq

    # Normalize the values in dict_cosine_sim
    dict_cosine_sim_normalized = min_max_normalize(dict_cosine_sim)

    # Sort and select the top 100 documents
    sorted_docs = sorted(dict_cosine_sim_normalized.items(), key=lambda x: x[1], reverse=True)
    top_100_docs = sorted_docs[:100]
    return top_100_docs


def search_res(inverted_title, inverted_title_2_words, inverted_body, inverted_anchor, query):
    query_filtered = tokenize_stemming(query)
    res_dict = defaultdict(float)

    if len(query_filtered) == 1:
        res_dict = score_one_word(inverted_title, inverted_body, inverted_title_2_words, inverted_anchor, query)

    if len(query_filtered) > 1:
        score_dic_res = score2(inverted_title, inverted_body, inverted_title_2_words, inverted_anchor, query)
        score_title_2_ngrams = search_title(query, inverted_title, inverted_title_2_words, ngram=True)
        
        for doc_id, score in score_dic_res.items():
            res_dict[doc_id] += score * 1/6

        for doc_id, score in score_title_2_ngrams:
            res_dict[doc_id] += score * 5/6

    # for doc_id, score in page_rank:
    #     res_dict[doc_id] += score


    sorted_docs = sorted(res_dict.items(), key=lambda x: x[1], reverse=True)
    top_100_docs = sorted_docs[:60]

    # Return a list of tuples containing the document ID and its title
    result = [(str(doc_id), inverted_title.doc_id_title[doc_id]) for doc_id, _ in top_100_docs]
    return result


# hepler functions for search_res
# query len = 1
def score_one_word(inverted_title, inverted_body, inverted_title_2_words, inverted_anchor, query):
    score_title = search_title(query, inverted_title, inverted_title_2_words, ngram=False)
    score_body = cosine_similarity(query, inverted_body)
    score_anchor = search_anchor(query, inverted_anchor)

    res_dict = defaultdict(float)
    for doc_id, score in score_title:
        res_dict[doc_id] += score * 7 / 16

    for doc_id, score in score_body:
        res_dict[doc_id] += score * 2 / 16

    for doc_id, score in score_anchor:
        res_dict[doc_id] += score * 7 / 16

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