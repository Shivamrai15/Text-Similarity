# Importing necessary libraries
import numpy as np
import re
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ------------------------------------------------------------------------------------#

# Creating a set of stopwords
stop_words = set(stopwords.words("english"))
# NLTK setup
porterStemmer = PorterStemmer()


# ------------------------------------------------------------------------------------#


# Function to preprocess the text data retrieved from text file
def preprocessingText(text):

    # Removing special characters and digits
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Converting text in lowercase and removing leading and trailing whitespace
    text = text.lower().strip()

    # Tokenizing, removing stopwords, and applying stemming
    tokens = [porterStemmer.stem(word) for word in text.split() if word not in stop_words]
    return " ".join(tokens)
# ------------------------------------------------------------------------------------#


# Loading pre-processed tfidf_matrix
tfidf_matrix = load_npz("tfidf_matrix.npz")


# ------------------------------------------------------------------------------------#


# Reading the inverted index text file
with open("inverted_index.txt", "r", encoding="utf-8") as file:
    inverted_idx_str = file.read()
    inverted_idx = eval(inverted_idx_str)


# ------------------------------------------------------------------------------------#


# Loading input queries
query_file = input("Enter the path of query text file\n")
try:
    with open(query_file, "r", encoding="utf-8") as file:
        queries = re.split(r"[*#]+", file.read())
except Exception as e:
    print("Query file not found")

# Removing the empty strings
filtered_documents = [text for text in queries if text.strip()]

# Performing the preprocessing of the filtered queries
preprocessed_queries = [preprocessingText(doc) for doc in filtered_documents]

# ------------------------------------------------------------------------------------#

# Processing preprocessed queries and calculating the cosine similarity
for i, query in enumerate(preprocessed_queries):


    # creating each document of the query file in a new array of given shape and type, filled with zeros.
    query_vector = np.zeros((1, len(inverted_idx)))
    # Splitting the query in each word
    words = query.split()

    for word in words:
        word = word.lower()
        if word in inverted_idx:
            query_vector[0, inverted_idx[word]] += 1

    # Calculating Cosine Similarity using sklearn
    cosine_similarity_scores = cosine_similarity(query_vector, tfidf_matrix)

    # Gettiing documents ids in decending order of their cosine similarity
    ranked_documents = np.argsort(cosine_similarity_scores[0])[::-1]

    # Printing the output ie similarity score
    print(f"\nQuery {i + 1} Output")
    for idx in ranked_documents:
        print(
            f"Document {idx + 1} - Similarity Score: {cosine_similarity_scores[0, idx]:.6f}"
        )
