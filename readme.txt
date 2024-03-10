## **Project Overview**

This project consists of several components aimed at building a search engine. Below is a brief description of each component:

### **1. Project_gcp.ipynb**

This notebook contains the code for index creation. It provides functionalities to create an index which is essential for efficient searching.

### **2. backend.py**

The `backend.py` file include all necessary functions for the search engine to process queries and return results. These functions include tokenization, stemming, and various ranking methods to ensure accurate and relevant search results.

### **3. inverted_index_gcp.py**

This Python script facilitates reading and writing of the index to Google Cloud Platform (GCP) storage bucket.

### **4. search_frontend.py**

The `search_frontend.py` file hosts the `search()` function. This function processes user queries and returns a list of up to 100 search results. The results are ordered from best to worst, ensuring that the most relevant content is presented to the user.
