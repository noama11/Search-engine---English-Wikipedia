# IR-final-project <br />
## Authors: Tomer Katzav, Noam Cohen, Shay Herling <br /> <br />


**Overview:**
This project consists of several components aimed at building a search engine for Wikipedia documents. This project is about creating a search engine for the entire Wikipedia corpus, comprising over 6,300,000 documents.

**Preprocessed the data:**
- Parsing
- Tokenization & Stemming
- Remove stopwords
- Separated every page into 3 components: title, body, anchor text.
- Created Inverted Index for Each component.

**Evaluation Metrics:**
- Precision@5: Number of relevant documents for a given query that are retrieved, divided by the number of retrieved documents.
- Recall: Number of relevant documents for a given query that are retrieved, divided by all relevant documents.
- F1@30: Average precision for a given query, averaging the averages for a benchmark query set.

**Inverted Index Details:**
- index_title 
- index_title_2_words_filter_0: Inverted Index with a filter > 0 in the nGrams method (n=2)
- index_anchor
- index_body_temp: In order to initialize document length, number of documents, and dictionary norm.
- index_body
- index_body_2_words: Inverted Index using the nGrams method (n=2)

## Project_gcp.ipynb

This notebook contains the code for index creation. It provides functionalities to create an index which is essential for efficient searching.

## backend.py

The `backend.py` file includes all necessary functions for the search engine to process queries and return results. These functions include tokenization, stemming, and various ranking methods to ensure accurate and relevant search results.

## inverted_index_gcp.py

This Python script facilitates reading and writing of the index to Google Cloud Platform (GCP) storage bucket.

## search_frontend.py

Flask app for the search engine frontend.
The `search()` function processes user queries and returns a list of up to 100 search results. The results are ordered from best to worst, ensuring that the most relevant content is presented to the user.

**Search / Main Route:**
Returns up to 100 of the best search results for the given query.
Our search engine employs advanced techniques including stopword removal, stemming, and inverted indexing to deliver highly relevant search results. By removing common stopwords and reducing words to their root form, we focus on meaningful terms in the corpus. Inverted indexing enables efficient retrieval of documents containing specific terms, facilitating accurate and fast search results.
