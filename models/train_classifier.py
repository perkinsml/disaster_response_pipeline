# Import libraries
import sys, re, pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from functools import partial
from sklearn.metrics import classification_report, make_scorer, fbeta_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import parallel_backend


def load_data(database_filepath: str) -> Tuple[pd.Series, np.array, list]:
    """ Load message and category data from the messages table in a SQLite
    database and return feature, target and category_name variables.

    Args:
    database_filepath: str.  File path to database file.

    Returns:
    X: Pandas Series.  A series of message text data.
    y: NumPy Array.  An array of message category data.
    category_names.  A list of category names.
    """

    # Load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages', con=engine)

    # Drop 'child_alone' column since this has 1 class only
    df.drop(columns=['child_alone'], axis=1, inplace=True)

    # Define feature, target and category_names and return them
    X = df.message
    y = df.iloc[:,4:].values
    category_names = df.iloc[:,4:].columns

    return X, y, category_names


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

def calculate_multioutput_f2(y_test: np.array, preds: np.array) -> float:
    """ Custom scoring function to calculate and return the mean binary
    F2 score across all categories

    Args:
    y_test: np.array.  Array of true feature test data.
    preds: np.arry.  Array of predicted feature test data.

    Returns:
    float.  Mean F2 score across all categories.
    """

    # Create a list of F2 scores across all categories
    score_list = []
    for i in range(y_test.shape[1]):
        score_list.append(fbeta_score(y_test[:,i], preds[:,i], \
                                      beta=2, average='binary', zero_division=0))

    # Return mean of F2 scores across all categories
    return np.mean(score_list)


def build_model() -> GridSearchCV:

    """ Return GridSearchCV object with a pipeline that includes a TfidfVectorizer
    transformation and a Support Vector Classifier within a linear kernel for
    prediction.

    The hyperparameter grid for the grid search is narrowed based on prior
    experimentation and the GridSearchCV object has a custom-defined scoring function.

    Args:
    None

    Returns:
    GridSearchCV object
    """

    # Define pipeline to fit and transform data with tfidf step and predict message categories
    # using a Support Vector Classifier with a linear kernel
    pipeline = Pipeline([('tfidf', TfidfVectorizer(tokenizer=partial(tokenize))),
                         ('clf', MultiOutputClassifier(LinearSVC(random_state=55,
                         max_iter=1000, dual=False)))],
                         verbose=True)

    # Define a grid of hyperparameters for the workflow
    # Note, this hyperparameter grid has been narrowed based on prior experimentation
    param_grid = {'tfidf__ngram_range': [(1,2)],
                  'tfidf__max_df': [0.7],
                  'tfidf__use_idf': [False],
                  'clf__estimator__C':np.logspace(-2, 5, 11)[5:7]}

    # Assign custom scorer as the 'calculate_multioutput_f2' function
    scorer = make_scorer(calculate_multioutput_f2, greater_is_better=True)

    # Define GridSearchCV object with custom scorer
    gs_cv = GridSearchCV(pipeline, param_grid, verbose=3, n_jobs=-1, scoring=scorer, \
                         cv=KFold(shuffle=True, random_state=55))

    return gs_cv

def evaluate_model(model: GridSearchCV, X_test: np.array, y_test: np.array, category_names:list):

    """ For each message category, display the classification report, baseline accuracy
    and model F2 score. At the end of the classification reports for each category,
    display the mean baseline accuracy, model accuracy, binary & weighted F1 & F2
    scores, and binary precision and recall scores across all categories.

    Args:
    model: GridSearchCV.  A fitted GridSearchCV instance.
    X_test: np.array.  Array of feature test data.
    y_test: np.array.  Array of outcome test data.
    category_names: list.  A list of category names.

    Returns:
    None
    """

    # Generate model predictions from best estimator
    preds = model.best_estimator_.predict(X_test)

    # Assign lists to store baseline accuracy, model accuracy, and binary
    # & weighted F1 & F2 scores for each category
    base_acc, model_acc, f1, f2, f1_w, f2_w, prec, recall = [], [], [], [], [], [], [], []

    # Cast y_test as a DataFrame for easier analysis
    y_test_df = pd.DataFrame(y_test, columns=category_names)

    # Iterate through each category
    for i in range(y_test.shape[1]):

        # Calculate additional evaluation metrics for each category and append to list
        base_acc.append(y_test_df[category_names[i]].value_counts().max() / len(y_test_df))
        model_acc.append(accuracy_score(y_test[:,i], preds[:,i]))
        f1.append(f1_score(y_test[:,i], preds[:,i], average='binary', zero_division=0))
        f2.append(fbeta_score(y_test[:,i], preds[:,i], beta=2, average='binary', zero_division=0))
        f1_w.append(f1_score(y_test[:,i], preds[:,i], average='weighted', zero_division=0))
        f2_w.append(fbeta_score(y_test[:,i], preds[:,i], beta=2, average='weighted', zero_division=0))
        prec.append(precision_score(y_test[:,i], preds[:,i], average='binary', zero_division=0))
        recall.append(recall_score(y_test[:,i], preds[:,i], average='binary', zero_division=0))

        # Display classification report, baseline accuracy and F2 score for each category
        print(f'Category: {category_names[i]}.  Baseline accuracy: {round(base_acc[i],4)}.  '
              f'F2 score: {round(f2[i],4)}.')
        print(classification_report(y_test[:,i], preds[:,i], digits=4, zero_division=0))
        print('-'*100)
        print()

    # Display mean evaluation metrics across all categories after the classification reports
    print('Mean evaluation metrics across all categories...\n')
    print(f'Mean baseline Accuracy: {round(np.mean(base_acc),4)}')
    print(f'Mean model Accuracy: {round(np.mean(model_acc),4)}')
    print(f'Mean binary F1: {round(np.mean(f1),4)}')
    print(f'Mean binary F2: {round(np.mean(f2),4)}')
    print(f'Mean weighted F1: {round(np.mean(f1_w),4)}')
    print(f'Mean weighted F2: {round(np.mean(f2_w),4)}')
    print(f'Mean binary Precision: {round(np.mean(prec),4)}')
    print(f'Mean binary Recall: {round(np.mean(recall),4)}\n')
    print('-'*100)


def save_model(model: GridSearchCV, model_filepath:str):
    """ Export model as pickle file to specified filepath.

    Args:
    model: GridSearchCV object.  Fitted model to be saved.
    model_filepath: str.  Destination path to save model to.

    Returns:
    None
    """

    # Export model as a pickle file
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, \
                                                            random_state=55)

        with parallel_backend('multiprocessing'):
            print('Building model...')
            model = build_model()

            print('Training model...')
            model.fit(X_train, y_train)

            print(model.best_score_)
            print(model.best_estimator_)

            print('Evaluating model...')
            evaluate_model(model, X_test, y_test, category_names)

            print('Saving model...\n    MODEL: {}'.format(model_filepath))
            save_model(model, model_filepath)

            print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
