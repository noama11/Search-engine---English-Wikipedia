# IR Project - Wikipedia Search Engine <br /> By Noam Cohen, Tomer Katzav and Shay Herling. <br /> <br />

# Project Overview

This project consists of several components aimed at building a search engine for Wikipedia documents. Below is a brief description of each component:

# Project_gcp.ipynb

This notebook contains the code for index creation. It provides functionalities to create an index which is essential for efficient searching.

# backend.py

The `backend.py` file include all necessary functions for the search engine to process queries and return results. These functions include tokenization, stemming, and various ranking methods to ensure accurate and relevant search results.

# inverted_index_gcp.py

This Python script facilitates reading and writing of the index to Google Cloud Platform (GCP) storage bucket.

# search_frontend.py

Flask app for search engine frontend.
search() function processes user queries and returns a list of up to 100 search results. The results are ordered from best to worst, ensuring that the most relevant content is presented to the user.
