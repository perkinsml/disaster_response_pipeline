# Table of Contents

1. [Background](#background)
1. [Project Overview](#project-overview)
1. [Installing](#installing)
    * [Dependencies](#dependencies)
    * [Executing the program](#executing-the-program)
    * [File Descriptions](#file-descriptions)
1. [Author](#author)
1. [License](#license)
1. [Acknowledgements](#acknowledgements)
1. [Web App Screenshots](#web-app-screenshots)

![web app header](https://github.com/perkinsml/disaster_response_pipeline/blob/master/web_app_header.png)

# Background
In a disaster situation, the crisis management response is typically spread across a range of organisations, with each one focussing on different components of the problem (e.g. water, aid, shelter, medical help, transport etc.).   The response of these organisations is informed by the millions and millions of communications – received either directly or via social media – that are typically generated during a disaster.  These communications are received at a time when organisations have the least capacity to manually filter through them to identify the priority communications that are relevant to their organisation’s response.

# Project Overview
This project aims to build a Natural Language Processing tool that can be used to automatically categorise messages received during a disaster situation. This project is part of Udacity’s Data Science Nanodegree program.

The dataset used to build this tool consists of tweets and text messages received during real-life disasters around the world.  These messages have been pre-labelled by Figure8 across a range of disaster response categories.  

An ETL and ML pipeline was used to build a supervised learning model to categorise these messages.  The project consists of 3 key parts:


1. **Data Processing**: an ETL pipeline to extract the messages and their categories from CSV files, clean the data and store the transformed dataset in a SQLite Database.
1. **Machine Learning**: a ML pipeline to train and evaluate a model to classify text messages into a range of categories
1. **Web App**: to provide an online tool that can be used to classify messages real-time

# Installing
Clone this GitHub repository:

```
git clone https://github.com/perkinsml/disaster_response_pipeline.git
```

You'll need to install the dr_utils package included in the repository by typing the command below in the root directory.  

```
pip install .
```

The dr_utils package includes a custom word tokenise function and a model scorer function - both of which are required to run the ML pipeline.  Given the class imbalance of the dataset and the priority of recall in this scenario, a custom f-beta scorer (with beta=2) was used to evaluate the model during grid search.  Please refer to the *ML Pipeline Preparation.ipynb* notebook for more detail.

## Dependencies
A list of dependencies is included in the requirements.txt file in this repository.
* Python 3.5+ (I used Python 3.7.6)
* Machine Learning Libraries: NumPy, SciPy, Pandas, Scikit-Learn
* Natural Language Process Libraries: NLTK
* SQLite Database Libraries: SQLAlchemy
* Web App and Data Visualization: Flask, Plotly
* Custom functions: dr-utils package (refer to installation instructions above)



## Executing the program
Run the following commands in the project's root directory to set up the database and model:
1. To run the ETL pipeline that extracts, cleans and transforms the message and category data, and stores it in the DisasterResponse.db database:
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
1. To run the ML pipeline that trains and saves the classifier:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

Running the ETL and ML pipelines is not necessary to start the web app.  

You can simply execute the following commands to run the web app., regardless of whether or not you choose to run the ETL and ML pipelines:
1. From the app directory:
    ```
    export FLASK_APP=run.py
    python -m flask run
    ```
1. Go to the link displayed in the command line.  For me, this is: http://127.0.0.1:5000/


## File Descriptions

<pre><code>
├── data
│   ├── disaster_messages.csv            # Dataset of messages
│   ├── disaster_categories.csv          # Dataset of message categories
│   ├── process_data.py                  # ETL pipeline script
|   ├── DisasterResponse.db              # SQLite database of message texts and categories
|   └── ETL Pipeline Preparation.ipynb   # Notebook demonstrating ETL pipeline script
|
├── models
│   ├── train_classifier.py              # ML pipeline script
|   ├── classifier.pkl                   # Pickled classification model
|   └── ML Pipeline Preparation.ipynb    # Notebook demonstrating ML pipeline build, train and test
|
├── dr_utils
|   └── custom_functions.py              # Custom functions used by  classification model
|
├── app
│   ├── run.py                           # Flask file that runs the app
│   └── templates
│       ├── master.html                  # Main page of web app
│       └── go.html                      # Classification results page of web app
│
├── requirements.txt                     # A list of required libraries and their versions
|
└── README.md
</code></pre>

# Author
[Matthew Perkins](https://github.com/perkinsml)

# License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Acknowledgements
* [Udacity](https://www.udacity.com/) for designing the Project
* [Figure8 (now known as Appen)](https://appen.com/) for providing the annotated dataset



# Web App Screenshots
Below is an example of the categorisation results displayed by the web app for the message:

>A massive fire has broken out after the storm. Homes are destroyed<br> and people have been left homeless.  We need doctors and clothing.

![results summary image](https://github.com/perkinsml/disaster_response_pipeline/blob/master/web_app_results_example.png)

The main page of the web app displays some visualisations of the message data provided by Figure Eight

![data charts image](https://github.com/perkinsml/disaster_response_pipeline/blob/master/data_overview.png)
