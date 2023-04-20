from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from flask import Flask, jsonify, request
import json

import pandas as pd
import numpy as np

import seaborn as sns

import nltk

path = './Reviews.csv'
df = pd.read_csv(path)

df = df.head(500)

def analyse(example):
    Tokens = nltk.word_tokenize(example)
    Tokens[:25]

    # nltk.download('averaged_perceptron_tagger')

    Tagged = nltk.pos_tag(Tokens)
    Tagged[:25]

    # nltk.download('maxent_ne_chunker')

    # nltk.download('words')

    entities = nltk.chunk.ne_chunk(Tagged)
    entities.pprint()


    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return  {
        'roberta_neg': float(scores[0]),
        'roberta_neu': float(scores[1]),
        'roberta_pos': float(scores[2])
    }


# nltk.download('punkt')

# example = 'This has an awesome taste'

api = Flask(__name__)



example =""
@api.route('/d', methods=['GET', 'POST'])
def d():
    body=request.json
    example= str(body['text'])
    dict=analyse(example)
    print (dict)
    
    response_body = {
        'Your Sentence': body['text'],
        'roberta_neg': dict['roberta_neg'],
        'roberta_neu': dict['roberta_neu'],
        'roberta_pos': dict['roberta_pos']
    }

    return response_body








@api.route('/')
def roberta():
    response_body = {
        'name': 'roberta',
    }

    return response_body


if __name__ =='__main__':
    api.run(host="0.0.0.0", port=5000)

