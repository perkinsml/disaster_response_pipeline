import json
import plotly
import pandas as pd
import numpy as np
import pickle
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
from sqlalchemy import create_engine

# Load custom scorer and tokenize functions from dr_utils package, installed
# as per README instructions
from dr_utils.custom_functions import calculate_multioutput_f2, tokenize


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)
# Drop 'child_alone' column, since model not trained for this category and it
# needs to be excluded from the category results displayed by the web app
df.drop(columns=['child_alone'], axis=1, inplace=True)

# Load model
model = pickle.load(open('../models/classifier.pkl', 'rb'))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # Extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    cat_ptgs = (100*df.iloc[:,4:].sum()/len(df)).sort_values(ascending=False)
    cat_names = list(cat_ptgs.index)

    non_cat_msg_length = df.loc[df.iloc[:,4:].sum(axis=1)==0,'message'].apply(len)
    cat_msg_length = df.loc[df.iloc[:,4:].sum(axis=1)>0,'message'].apply(len)

    # Create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker_color='#14A64E'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'height':400,
                'width':1000,
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_ptgs,
                    marker_color='#F67D04'
                )
            ],

            'layout': {
                'title': 'Percentage of messages assigned to each category',
                'height':500,
                'width':1000,
                # 'margin':{'b':150, 'l':200},
                'yaxis': {
                    'title': "Percentage (%)"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle':30
                }
            }
        },

        {
            'data': [
                Histogram(
                    x=cat_msg_length,
                    name='Categorised messages',
                    marker_color='#8C11F2',
                    # opacity=0.4
                ),

                Histogram(
                    x=non_cat_msg_length,
                    name='Non-categorised messages',
                    marker_color='#3BB0CC',
                    # opacity=0.4
                ),
            ],

            'layout': {
                'title': 'Histogram of message length (trucated at 500 characters)',
                'height':600,
                'width':1000,
                'barmode':'overlay',
                # 'margin':{'b':150, 'l':200},
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Length",
                    'range':[0,500]
                }
            }
        }
    ]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# Web page that handles user query and displays model results
@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '')

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html. Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
