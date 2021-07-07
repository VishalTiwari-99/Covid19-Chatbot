from flask import Flask, render_template, url_for, request, redirect
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired
import os
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
import json
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'my_sec_key'

#####--database setup-----##########
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///'+os.path.join(basedir,'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
Migrate(app,db)

class QnA(db.Model):
    __tablename__ = 'list_qna'
    id = db.Column(db.Integer,primary_key=True)
    data = db.Column(db.Text)
    def __init__(self,data):
        self.data = data
    def __repr__(self):
        return f"{self.data}"

class userform(FlaskForm):
    query = TextAreaField(validators=[DataRequired()])
    submit = SubmitField('Ask')
class askform(FlaskForm):
    movie = TextAreaField(validators=[DataRequired()])
    submit = SubmitField('Ask')


############=----movie Recommendation----##################
df2 = pd.read_csv('./model/tmdb.csv')

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])
all_titles = [df2['title'][i] for i in range(len(df2['title']))]


def get_recommendations(title):
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    tit = df2['title'].iloc[movie_indices]
    dat = df2['release_date'].iloc[movie_indices]
    return_df = pd.DataFrame(columns=['Title', 'Year'])
    return_df['Title'] = tit
    return_df['Year'] = dat
    return return_df




def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model, words, classes):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.1
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg, model, intents, words, classes):
    ints = predict_class(msg, model, words, classes)
    res = getResponse(ints, intents)
    return res



@app.route('/',methods=['GET','POST'])
def index():
    form = userform()
    question=""
    response = ""
    conversation = []
    if form.validate_on_submit():
        question = form.query.data
        new_data = QnA(question)
        db.session.add(new_data)
        db.session.commit()
        model = load_model('chatbot_model.h5')
        intents = json.loads(open('intents.json',encoding="utf8").read())
        words = pickle.load(open('words.pkl','rb'))
        classes = pickle.load(open('classes.pkl','rb'))
        response = chatbot_response(question, model, intents, words, classes)
        new_data = QnA(response)
        db.session.add(new_data)
        db.session.commit()

        conversation = QnA.query.all()

    return render_template('index.html',form=form,question=question,response=response,conversation=conversation)

@app.route('/infomation',methods=['GET','POST'])
def info():
    return render_template('info.html')

@app.route('/recommend', methods=['GET', 'POST'])
def main():
    form = askform()
    m_name = ""
    names = []
    if form.validate_on_submit():
        m_name = form.movie.data
        result_final = get_recommendations(m_name)
        for i in range(len(result_final)):
            names.append(result_final.iloc[i][0])

    return render_template('index2.html',form=form,names=names)



if __name__ == '__main__':
    app.run(debug=True)
