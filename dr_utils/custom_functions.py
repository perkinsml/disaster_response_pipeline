import re
import numpy as np
from sklearn.metrics import fbeta_score
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Define custom scoring function for multiple outputs
def calculate_multioutput_f2(y_test, preds, beta=2):
    """ Custom scoring function to calculate and return the mean binary
    F2 score across all categories

    Args:
    y_test: np.array.  Array of true feature test data.
    preds: np.arry.  Array of predicted feature test data

    Returns:
    float.  Mean F2 score across all categories.
    """

    # Create a list of F2 scores across all categories
    score_list = []
    for i in range(y_test.shape[1]):
        score_list.append(fbeta_score(y_test[:,i], preds[:,i], \
                                      beta=beta, average='binary', zero_division=0))

    # Return mean of F2 scores across all categories
    return np.mean(score_list)

# Define custom function to process a text string into normalised tokens
def tokenize(text: str) -> list:
    """ Normalise and tokenise a text string, remove stop words and return
    list of tokens after lemmatisation and stemming.

    Args:
    text: str.  A string of text to be processed.

    Returns:
    list. A list of processed tokens.
    """

    # Normalise text by removing capitalisation and punctuation.
    # Replace non-alpha-numeric characters with a space to avoid
    # incorrect concatenation of words within text.
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

    # Tokenise text
    tokens = word_tokenize(text)

    # Remove stop words and lemmatise text
    nltk_stop_words = stopwords.words('english')
    tokens = [WordNetLemmatizer().lemmatize(token.strip()) for token in tokens \
             if token.strip() not in nltk_stop_words]

    # Finally apply Stemming
    tokens = [PorterStemmer().stem(token) for token in tokens]

    return tokens
