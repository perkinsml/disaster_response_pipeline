# Overview
In a disaster situation, the crisis management response is typically spread across a range of organisations, with each one focussing on different components of the problem (e.g. water, aid, shelter, transport etc.).   The response of these organisations is informed by the millions and millions of communications – received either directly or via social media – that are typically generated during a disaster.  These communications are received at a time when organisations have the least capacity to manually filter through them to identify the priority communications that are relevant to their organisation’s response.

This project aims to build a Natural Language Processing tool that can be used to automatically categorise the messages received during a disaster situation.

This project is part of Udacity’s Data Science Nanodegree program.  The dataset used to build this tool consists of tweets and text messages received during real-life disasters around the world.  These messages have been pre-labelled by Figure8 across a range of disaster response categories.  

An ETL and ML pipeline was used to build a supervised learning model to categorise these messages and the project consists of 3 key parts:


1. **Data Processing**: an ETL pipeline to extract the messages and their categories from CSV files, clean the data and store the dataset in a SQLite Database.
1. **Machine Learning**: a ML pipeline to train and evaluate a model to classify test messages in categories
1. **Web App**: to provide an online tool that can be used to classify messages real-time

# Installation

## File Descriptions

<pre><code>
├── app
│   ├── run.py                           # Flask file that runs the app
│   └── templates
│       ├── master.html                  # Main page of web app
│       └── go.html                      # Classification results page of web app
|
├── data
│   ├── disaster_messages.csv            # Dataset of messages
│   ├── disaster_categories.csv          # Dataset of message categories
│   ├── process_data.py                  # ETL pipeline script to read message data from CSV, clean it and store it in the DisasterReponse.db database
|   ├── DisasterResponse.db              # SQLite database of cleaned message texts and their categories
|   └── ETL Pipeline Preparation.ipynb   # A notebook demonstrating the implemented ETL pipeline script step-by-step
|
├── models
│   ├── train_classifier.py              # Train ML model
|   ├── classifier.pkl                   # Pickled classification model built using the train_classifier.py script
|   └── ML Pipeline Preparation.ipynb    # A notebook demonstrating the ML pipeline build, train, test and evaluate process
|
├── dr_utils
|   └── custom_functions.py              # A script with custom functions used by the classification model
|
└── README.md
</code></pre>

# Web App Screenshots
Below is an example of the categorisation results by the web app for the message:
> A massive fire has broken out after the storm. Homes are destroyed and people have been left homeless.  We need doctors and clothing.
!('web_app_results_example.png')
