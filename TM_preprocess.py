#import packages
import pandas as pd 
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
import nltk
from nltk.corpus import stopwords
import mysql.connector
import string
from nltk.corpus import words
import enchant
import spacy
import re
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import os

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 
import datetime
import tqdm
from gensim.test.utils import datapath
import numpy as np
# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
nltk.download('stopwords')
from tmtoolkit import topicmod
import math

#functions to read in and preprocess the data
#create database connection
mydb = mysql.connector.connect(user='root',
                              password = 'SharonF1.',
                              host='localhost',
                              port = '3306',
                              database = 'slackdatabase',
                              auth_plugin='mysql_native_password',
                              raw = False
)

c = mydb.cursor(buffered=True)

#phase dictionary dict[year] = dict[keys = phase, values = list [start_date, end_date]]

phases = {} 

dates_19 = {"3ideas": [datetime.datetime(2019, 9, 1), datetime.datetime(2019, 9, 23)], "sketchModel": [datetime.datetime(2019, 9, 24), datetime.datetime(2019, 10,3)], "mockupReview": [datetime.datetime(2019, 10, 4), datetime.datetime(2019, 10, 17)], "FinalSelection": [datetime.datetime(2019, 10, 18), datetime.datetime(2019, 10, 21)], "AssemblyReview": [datetime.datetime(2019, 10,22), datetime.datetime(2019, 11, 1)], "TechnicalReview": [datetime.datetime(2019, 11, 2), datetime.datetime(2019, 11, 14)], "FinalPresentation": [datetime.datetime(2019, 11, 15), datetime.datetime(2019, 12, 9)]}
phases[2019] = dates_19

dates_16 = {"3ideas": [datetime.datetime(2016, 9, 1), datetime.datetime(2016, 9, 26)], "sketchModel": [datetime.datetime(2016, 9, 27), datetime.datetime(2016, 10,6)], "mockupReview": [datetime.datetime(2016, 10, 7), datetime.datetime(2016, 10, 20)], "FinalSelection": [datetime.datetime(2016, 10, 21), datetime.datetime(2016, 10, 24)], "AssemblyReview": [datetime.datetime(2016, 10,25), datetime.datetime(2016, 11, 4)], "TechnicalReview": [datetime.datetime(2016, 11, 5), datetime.datetime(2016, 11, 17)], "FinalPresentation": [datetime.datetime(2016, 11, 18), datetime.datetime(2016, 12, 12)]}
phases[2016] = dates_16

dates_17 = {"3ideas": [datetime.datetime(2017, 9, 1), datetime.datetime(2017, 9, 25)], "sketchModel": [datetime.datetime(2017, 9, 26), datetime.datetime(2017, 10,5)], "mockupReview": [datetime.datetime(2017, 10, 6), datetime.datetime(2017, 10, 19)], "FinalSelection": [datetime.datetime(2017, 10, 20), datetime.datetime(2017, 10, 23)], "AssemblyReview": [datetime.datetime(2017, 10,24), datetime.datetime(2017, 11, 3)], "TechnicalReview": [datetime.datetime(2017, 11, 4), datetime.datetime(2017, 11, 16)], "FinalPresentation": [datetime.datetime(2017, 11, 17), datetime.datetime(2017, 12, 11)]}
phases[2017] = dates_17

dates_18 = {"3ideas": [datetime.datetime(2018, 9, 1), datetime.datetime(2018, 9, 24)], "sketchModel": [datetime.datetime(2018, 9, 25), datetime.datetime(2018, 10,4)], "mockupReview": [datetime.datetime(2018, 10, 5), datetime.datetime(2018, 10, 18)], "FinalSelection": [datetime.datetime(2018, 10, 19), datetime.datetime(2018, 10, 22)], "AssemblyReview": [datetime.datetime(2018, 10,23), datetime.datetime(2018, 11, 2)], "TechnicalReview": [datetime.datetime(2018, 11, 3), datetime.datetime(2018, 11, 15)], "FinalPresentation": [datetime.datetime(2018, 11, 16), datetime.datetime(2018, 12, 10)]}
phases[2018] = dates_18


def getPhaseDates(phase, year):
    '''returns the dates for the phase and the year'''

    return phases[year][phase]


def readTeamData (team, phase, year):
    ''' read data from database for team, phase, year'''
    start_date = getPhaseDates(phase, year)[0]
    end_date = getPhaseDates(phase, year)[1]
    
    query = "SELECT * FROM message m INNER JOIN user u ON m.id_user = u.slack_uid INNER JOIN team t on u.id_team = t.id INNER JOIN channel c on c.slack_cid = m.id_channel WHERE year(m.timestamp) = '%s' AND t.name = '%s' AND DATE(m.timestamp) >= '%s' AND DATE(m.timestamp) <= '%s' " % (year, team, start_date, end_date) 
    c.execute(query)
    df = pd.DataFrame(c.fetchall())

    if len(df) > 0: 
        df.columns = c.column_names

    return df 

def readAllData (phase): 
    '''read data from database for all teams over all years, per phase'''
    
    dfs = [] 
    
    for year in [2016, 2017, 2018, 2019]: 
        start_date = getPhaseDates(phase, year)[0]
        end_date = getPhaseDates(phase, year)[1]
        query = "SELECT * FROM message m INNER JOIN user u ON m.id_user = u.slack_uid INNER JOIN team t on u.id_team = t.id INNER JOIN channel c on c.slack_cid = m.id_channel WHERE year(m.timestamp) = '%s' AND DATE(m.timestamp) >= '%s' AND DATE(m.timestamp) <= '%s' " % (year, start_date, end_date) 
        c.execute(query)
        df = pd.DataFrame(c.fetchall())
        df.columns = c.column_names
        dfs.append(df)
    
    all_data = pd.concat(dfs)

    return all_data

def readTotalData (): 
    '''read data from database for all teams, all years, all phases'''
    
    query = "SELECT * FROM message m INNER JOIN user u ON m.id_user = u.slack_uid INNER JOIN team t on u.id_team = t.id INNER JOIN channel c on c.slack_cid = m.id_channel"
    c.execute(query)
    df = pd.DataFrame(c.fetchall())
    df.columns = c.column_names

    return df

def generalPreprocessing(data):
    #take out @'s and <channels!>

    data['comments_processed'] = data['comment'].map(lambda x: re.sub(r'\<(.*[^>])>',r"", x))
 
    #remove punctuation
    translator = str.maketrans({key: None for key in string.punctuation +"123456789-"})
    data['comments_processed'] = data['comments_processed'].map(lambda x: x.translate(translator)) #takes out punctuation 

    #lowercase
    data['comments_processed'] = data['comments_processed'].map(lambda x: x.lower())
    data['date'] = data['timestamp'].map(lambda x: x.date())


    return data


def createChannelDocs (data):
    '''creates the documents for the scenario where each channel is its own document''' 

    #each channel is its own document
    comments_per_channel = []
    for i in data['id_channel'].unique():
        channel_comment = data[data['id_channel']==i]
        channel_comment = list(channel_comment['comments_processed'].values)
        doc = " ".join(channel_comment)
        comments_per_channel.append(doc) 
    
    return comments_per_channel

def createChannelDayDocs(data):
    '''creates the documents for the scenario where each channel-day is its own document'''
    comments_per_channel_per_day = []
    for i in data['id_channel'].unique():
        #print(i)
        for j in data['date'].unique():
            doc = ""
            day_comment = data.loc[(data['id_channel']==i) & (data['date']==j)]
            day_comment = list(day_comment['comments_processed'].values)
            doc = " ".join(day_comment)
            comments_per_channel_per_day.append(doc)
    
    #write to excel file to save - can uncomment this if working with large portions of data- this saves it to a CSV to speed it up if you're running it multiple times

    # df = pd.DataFrame(comments_per_channel_per_day)
    # df.to_csv('comments per channel per day.csv', index=False)
    # print("csv written")

    return comments_per_channel_per_day

def createTeamPhaseDocs(data):
    '''creates documents for the scenario where each team-phase is a document - assume each input is already one phase'''
    comments_per_team = [] 
    teams = []
    for i in data['id_team'].unique():
        teams.append(i)
        team_comment = data[data['id_team']==i]
        team_comment = list(team_comment['comments_processed'].values)
        doc = " ".join(team_comment)
        comments_per_team.append(doc) 
    return comments_per_team

def createPersonPhaseDocs(data):
    '''creates documents for the scenario where each person is a document, assume each input is already one phase'''
    comments_per_person = [] 
    people = []
    for i in data['id_user'].unique():
        people.append(i)
        person_comment = data[data['id_user']==i]
        person_comment = list(person_comment['comments_processed'].values)
        doc = " ".join(person_comment)
        comments_per_person.append(doc) 
    return comments_per_person
    
def createMessageDocs(data):
    '''creates documents for the scenario where each message is a document (STTM) - each input is a team/phase'''
    data = data.loc[data['comments_processed'].str.len() >0] 
    docs = data['comments_processed'].tolist() 
    return docs


def sentence_to_words(sentences): 
    '''converts a document into a list of lowercase tokens, ignoring tokens that are too short or too long'''
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc= True, ))

def remove_stopwords(texts):
    '''removes stopwords in the nltk package'''
    stop_words = stopwords.words('english')
    stop_words.extend(['from'])
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    nlp = spacy.load('en', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def buildBigramTrigram(data_words):
    '''builds bigrams (two word phrases) and trigrams (three word phrases) that normally appear together using the gensim package'''
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    return bigram_mod, trigram_mod

def make_bigrams(bigram_mod, texts):
    '''make bigrams using the gensim package'''
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(trigram_mod, bigram_mod,  texts):
    '''make trigrams using the gensim package'''
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def extendedPreprocessing(documents):
    '''after the documents have been prepared, this is where you stem and lemmatize the values, lowecase, take out large or small words and check spelling
    1) remove common stop words
    2) correct misselled words
    3) remove words other than nouns, verbs, adjectives and adverbs
    4) unify words to their basic form (lemmatization)
    5) remove very frequent words - maybe even words from the project description
    6) find bigrams and trigrams
    '''

    #prepare data for LDA

    #tokenize words
    data_words = list(sentence_to_words(documents))
    
    data_words = remove_stopwords(data_words)

    #check spelling of words 
    d = enchant.Dict("en_US")

    data_check = [[d.check(x) for x in doc] for doc in data_words]
    for i in range(0, len(data_words)): 
        data_words[i] = [j for (j,v) in zip(data_words[i], data_check[i]) if v]

    
    bigram_mod, trigram_mod = buildBigramTrigram(data_words)
    data_words_bigrams = make_bigrams(bigram_mod, data_words)
    data_words_trigrams = make_trigrams(trigram_mod, bigram_mod, data_words)
    
    #lemmatize and keep only nouns, verbs, adjectives and adverbs
    nlp = spacy.load('en', disable=['parser', 'ner'])
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    #remove very frequent words
    frequent_words = ['yeah', 'sorry', 'class', 'think', 'need', 'would', 'get', 'like','also', 'go', 'put', 'use', 'want', 'let', 'do']
    data_specific = [[word for word in simple_preprocess(str(doc)) if word not in frequent_words] for doc in data_lemmatized]
 
    #create the corpus and dictionary

    # Create Dictionary
    id2word = corpora.Dictionary(data_specific)

    # Create Corpus
    texts = data_specific

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    return id2word, texts, corpus