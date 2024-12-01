# text preprocessing modules
import pandas as pd
import numpy as np 
from string import punctuation
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import os
from os.path import dirname, join, realpath
import joblib
import uvicorn
from fastapi import FastAPI 

class PredictTopicOutput:
    def __init__(self):
        self.numberOfTopics = None
        self.Topic1 = None
        self.Topic2 = None
        self.Topic3 = None
        self.probability = None

    def __str__(self):
        return f"Number of Topics: {self.numberOfTopics}\n" \
               f"Topic 1: {self.Topic1}\n" \
               f"Topic 2: {self.Topic2}\n" \
               f"Topic 3: {self.Topic3}\n" \
               f"Probability: {self.probability}"

app = FastAPI(
    title="Bertopic model ",
    description="A simple API that use NLP model to predict topics",
    version="0.1",
)

# load the sentiment model
with open(
    join(dirname(realpath(__file__)), "Bertopic_model_cpu.pkl"), "rb"
) as f:
    model = joblib.load(f)

# cleaning the data
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers
    
    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])
        
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
        
    # Return a list of words
    return text

@app.get("/predict-review")
def predict_topics(review: str):
    """
    A simple function that receive a  content and predict the topic of the content.
    :param review:
    :return: prediction, probabilities
    """
    # clean the review
    cleaned_review = text_cleaning(review)
    
    # perform prediction
    num_of_topics = 3
    similar_topics, similarity = model.find_topics(cleaned_review, top_n=num_of_topics)
        #return similar_topics, similarity
    topics_name=pd.read_excel("topic_list_cpu.xlsx")
    topic_dict = topics_name.set_index("Topic")["Representation"].to_dict()
    
    predictTopicOutput = PredictTopicOutput()
    predictTopicOutput.numberOfTopics = num_of_topics
    predictTopicOutput.Topic1 = f'Topic Number: {str(similar_topics[0])} Associated Words:{topic_dict[similar_topics[0]]}'
    predictTopicOutput.Topic2 = f'Topic Number: {str(similar_topics[1])} Associated Words:{topic_dict[similar_topics[1]]}'
    predictTopicOutput.Topic3 = f'Topic Number: {str(similar_topics[2])} Associated Words:{topic_dict[similar_topics[2]]}'
    predictTopicOutput.probability = str(np.round(similarity,2))
    
    return predictTopicOutput
    
   
# Start the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)