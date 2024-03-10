**IR-final-project**
Authors:
Tomer Katzav, Noam Cohen, Shay Herling

Overview:
This project consists of several components aimed at building a search engine for Wikipedia documents.This project is about creating a search engine to the entire Wikipedia corpus, above 6,300,000 documents

Preprocessed the data-

Parsing
Tokenization & Stemming
remove stopwords
Separated every page to 3 components: title, body, anchor text.
created Inverted Index to Each component.

Evaluation- 
The evaluation metrics we used are:
Precision@5: number of relevant documents for given query that retrived, divide by number of retrived documents.
Recall: number of relevant documents for given query that retrived, divide by all relevant documents.
F1@30: Find the avg precision for a given query, averaging the averages for a benchmark query set.

Inverted Index Details-
index_title 
index_title_2_words_filter_0 : Inverted Index with filter > 0 in method nGrams (n=2)
index_anchor
index_body_temp : in order to initlizied doc lenght, Number of docs and dict norm 
index_body
index_body_2_words : Inverted Index in method nGrams (n=2)

# Project_gcp.ipynb

This notebook contains the code for index creation. It provides functionalities to create an index which is essential for efficient searching.

# backend.py

The `backend.py` file include all necessary functions for the search engine to process queries and return results. These functions include tokenization, stemming, and various ranking methods to ensure accurate and relevant search results.

# inverted_index_gcp.py

This Python script facilitates reading and writing of the index to Google Cloud Platform (GCP) storage bucket.

# search_frontend.py

Flask app for search engine frontend.
search() function processes user queries and returns a list of up to 100 search results. The results are ordered from best to worst, ensuring that the most relevant content is presented to the user.

Serach / main route:
Returns up to a 100 of the best search results for the given query.
Our search engine employs advanced techniques including stopword removal, stemming, and inverted indexing to deliver highly relevant search results. By removing common stopwords and reducing words to their root form, we focus on meaningful terms in the corpus.  Inverted indexing enables efficient retrieval of documents containing specific terms, facilitating accurate and fast search results.
