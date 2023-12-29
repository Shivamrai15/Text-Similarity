
## **Installation**

Install necessary packages with pip

```bash
  pip install scikit-learn
  pip install nltk
  pip install scipy
```
    
# **Processing File**
This program computes a TF-IDF Weighted Term document Incident matrix and a text file containing an Inverted Index.


### Program Details
The program first reads the text files stored in the files directory of the project and then preprocess the document by removing special characters, stopwords and stemming. After that it computes the TF-IDF Weighted Term document Incident matrix and store it in the project root directory in a npz (NumPyZipped) file. It builds a text file containing the inverted index of terms and their corresponding document.


### Prerequisites
```bash
Python 3
os
re
scipy
nltk
sklearn
```


### Usage
Place input text files in the files directory of the root of project before executing the preprocess.py file program

To run the program, open a terminal window and navigate to the projects's directory and use the command
```bash
python preprocess.py
```

### Output
Output of this program is a npz file containing TF-IDF Weighted Term document Incident matrix and a text file containing an Inverted Index.


# **Query File**
This program calculates and prints the cosine similarity score of the relevant documents on the console or terminal.

### Prerequisites
```bash
Python 3
re
numpy
scipy
sklearn
```

### Usage

To run the program, open a terminal window and navigate to the projects's directory and use the command
```bash
python query.py
```
After that the program will ask the path of the query file,so enter the query file path


### Output
The program will print the cosine similarity between the two documents in decreasing order of their cosine similarity. 









