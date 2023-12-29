# Importing necessary libraries

import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.sparse import save_npz, csr_matrix

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


# Function to read documents from a text file
def readDocuments(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()

    # Splitting the data based on consecutive asterisks
    text = re.split(r'[*#]+', data)
    return text


# ------------------------------------------------------------------------------------#


# Function to create inverted index
def createTDIFMatrixAndInvertedIndex(documents):
    # Using TfidfVectorizer to compute TF-IDF matrix
    vector = TfidfVectorizer()
    tfidf_matrix = vector.fit_transform(documents)

    inverted_index = {}
    for term, index in vector.vocabulary_.items():
        inverted_index[term] = index
    return tfidf_matrix, inverted_index


# ------------------------------------------------------------------------------------#


# Main function
def main():
    # Read documents from files
    folder_name = "files"
    documents = []
    for file_name in os.listdir(folder_name):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_name, file_name)
            documents.extend(readDocuments(file_path))

    # Removing the empty strings
    filtered_documents = [text for text in documents if text.strip()]

    # Preprocessing documents
    preprocessed_documents = [preprocessingText(doc) for doc in filtered_documents]


    # Creating TF-IDF matrix and inverted index
    tfidf_matrix, inverted_index = createTDIFMatrixAndInvertedIndex(preprocessed_documents)

    print("Preprocessing step is completed\n############################################")
    print("\nTFIDF Matrix\n")
    print(tfidf_matrix)
    print("\n############################################")
    print("Inverted index is created\n")
    print("Click below to visit the inverted index file\n", '\\inverted_index.txt')

    # print("TFIDF Matrix\n", tfidf_matrix)
    save_npz('tfidf_matrix.npz', csr_matrix(tfidf_matrix))

    with open("inverted_index.txt", "w", encoding="utf-8") as file:
        file.write(str(inverted_index))



if __name__ == "__main__":
    main()