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
import pickle 
import GSDMM

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
from mgp import *
warnings.filterwarnings("ignore",category=DeprecationWarning)
nltk.download('stopwords')
from tmtoolkit import topicmod
import math

from TM_preprocess import *

#load initial DB connection and dates
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

years = [2016, 2017, 2018, 2019]
phases_2 = ["3ideas", "sketchModel", "mockupReview", "FinalSelection", "AssemblyReview", "TechnicalReview", "FinalPresentation"]
teams = ["blue", "green", "orange", "pink", "purple", "red", "silver", "yellow"]

###NOTES/RESOURCES/REFERENCES#####:

#GDSMM algorithm: https://github.com/rwalk/gsdmm
#short text topic modelling tutorial: https://github.com/Matyyas/short_text_topic_modeling/blob/master/notebook_sttm_example.ipynb
##other STTM that I didn't end up using for this one, but also could be useful in the future: https://github.com/jrmazarura/GPM 
##explanation of coherence measurements: https://markroxor.github.io/gensim/static/notebooks/topic_coherence_tutorial.html
#exploring the space of topic coherence measures paper - associated github page: https://github.com/dice-group/Palmetto
#tmtoolkit: https://tmtoolkit.readthedocs.io/en/latest/api.html?highlight=metric_coherence#tmtoolkit.topicmod.evaluate.metric_coherence_gensim
#Gensim coherence model: https://radimrehurek.com/gensim/models/coherencemodel.html
#unused but other STTM with coherence measure in JAVA: https://github.com/qiang2100/STTM/blob/master/src/eval/CoherenceEval.java 
#Kaggle STTM competition: https://www.kaggle.com/ptfrwrd/gsdmm-short-text-clustering/code , https://www.kaggle.com/ptfrwrd/gsdmm-short-text-clustering
#reddit, stackoverflow post for why UMass is negative: https://www.reddit.com/r/learnmachinelearning/comments/9bcr77/coherence_score_u_mass/e52b8jw/ https://stackoverflow.com/questions/62032372/coherence-score-u-mass-18-is-good-or-bad
#how gensim calculates coherence: https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/topic_coherence/aggregation.py
#code for the "exploring the space of topic model coherence" paper: https://github.com/dice-group/Palmetto
#Gensim topic coherence tutorial: https://markroxor.github.io/gensim/static/notebooks/topic_coherence_tutorial.html
#good NLP github: https://github.com/shuyo/iir 
#STTM GDSMM medium implementation tutorial: https://towardsdatascience.com/short-text-topic-modeling-70e50a57c883 
#visualization: https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
#original LDA tutorial (some code from here): https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
#Gensim tutorial: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

def train_STTM(k, alpha, beta, n_iters, docs, file_name):
    '''train the mgp STTM, https://github.com/rwalk/gsdmm'''
    mgp = MovieGroupProcess(K=k, alpha=alpha, beta=beta, n_iters=n_iters)
    vocab = set(x for doc in docs for x in doc)
    n_terms = len(vocab)
    y, convergence_data = mgp.fit(docs, n_terms)
    # Save model - can uncomment this if you want to save the model 
    # with open(file_name, “wb”) as f:
    # pickle.dump(mgp, f)
    # f.close()

    return mgp, y, convergence_data


def top_words(distribution, top_index, num_words):
    '''returns the top_index number of words for each topic, https://github.com/Matyyas/short_text_topic_modeling/blob/master/notebook_sttm_example.ipynb'''
    topic_df = pd.DataFrame(columns = ['Topic', "words"])
    for topic in top_index:
        pairs = sorted([(k, v) for k, v in distribution[topic].items()], key=lambda x: x[1], reverse=True)
        print(f"Cluster {topic} : {pairs[:num_words]}")
        print('-'*30)
        topic_df = topic_df.append({'Topic': topic, 'words': pairs[:num_words]}, ignore_index = True)
    return topic_df


def show_topics(mgp):
    '''function to print out the top words per topic as calculated in top_words'''
    doc_count = np.array(mgp.cluster_doc_count)
    print('Number of documents per topic :', doc_count)
    print('*'*20)
    # Topics sorted by the number of document they are allocated to
    top_index = doc_count.argsort()[-len(doc_count):][::-1]
    print(top_index)
    print('Most important clusters (by number of docs inside):', top_index)
    print('*'*20)
    # Show the top 10 words in term frequency for each cluster 
    topic_df = top_words(mgp.cluster_word_distribution, top_index, 20)
    return topic_df

def psu_coh(mgp, y, texts):
    '''my own measure of how "certain" the algorithm is of the topic assignment of each message. Averages the percent that the message "belongs" to each topic, for every message in a topic. Closer to 1 is better'''
    coh = {} 
    for text in texts:
        topic, score = mgp.choose_best_label(text)
        if topic not in coh.keys():
            coh[topic] = [score]
        else:
            coh[topic].append(score)
    coh_per_topic = [] 
    for top in coh.keys():
        avg_coh = np.mean(coh[top])
        coh_per_topic.append(avg_coh)

    return np.mean(coh_per_topic)

def coherence_umass(topic_dist, texts):
    '''uMass coherence metric. General formula adapted from https://github.com/jrmazarura/GPM which implements a non-normalized version of this. 
    Formula is from the paper "Exploring the space of topic coherence metrics" by Michael Roder. Written using loops so ia quite slow on large datasets - consider vectorizing if meant to be used more. INTRINSIC COHERENCE''' 
    #turn words per topic dictionary into a list of lists
    words_per_topic = []
    for topic in topic_dist:
        topic_words = []
        if len(topic) > 1:
            for word in topic.keys():
                topic_words.append(word)
            words_per_topic.append(topic_words)
                

    coherence_corpus=[]
    for a in texts:
        coherence_doc=[]
        for b in a:
            coherence_doc.append(b)
        coherence_corpus.append(coherence_doc)
    
    nTopics = len(words_per_topic)

        #number of top words per topic
    nDocs = len(coherence_corpus) #number of documents
    epsilon = 1 #smoothing parameter
       
    
    coherence= []       
    for t in range(0,nTopics): #calculate coherence
        nTopWords = len(words_per_topic[t]) 
        coherence_per_topic = []            
        for vj in range(1, nTopWords): #each word in topic t
                
            Dvj = 0
            for d in range(0,nDocs): 
                if (words_per_topic[t][vj] in coherence_corpus[d]): #check how many docs contain word vj
                    Dvj += 1
                    
            for vi in range(0, vj): 
                Dvjvi = 0
                Dvi= 0
                for d in range(0,nDocs): 
                    
                    if (words_per_topic[t][vj] in coherence_corpus[d]) and (words_per_topic[t][vi] in coherence_corpus[d]): #check how many docs contain both word vj and vi
                        Dvjvi += 1
                    if words_per_topic[t][vi] in coherence_corpus[d]:
                        Dvi+=1
                
            
                coherence_per_topic.append(math.log(Dvjvi+epsilon/(float(Dvj)),10))
        coherence.append((2/(nTopWords*(nTopWords-1)))*np.sum(coherence_per_topic))        
    print("average topic: ", (np.sum(coherence))/nTopics)

    return (np.sum(coherence))/nTopics

def coherence_uci(topic_dist, texts):
    '''UCI coherence metric. General formula adapted from https://github.com/jrmazarura/GPM which implements a non-normalized version of this. 
    Formula is from the paper "Exploring the space of topic coherence metrics" by Michael Roder. Written using loops so ia quite slow on large datasets - consider vectorizing if meant to be used more. INTRINSIC COHERENCE''' 
    #turn words per topic dictionary into a list of lists
    words_per_topic = []
    for topic in topic_dist:
        topic_words = []
        if len(topic) > 1:
            for word in topic.keys():
                topic_words.append(word)
            words_per_topic.append(topic_words)
                

    coherence_corpus=[]
    for a in texts:
        coherence_doc=[]
        for b in a:
            coherence_doc.append(b)
        coherence_corpus.append(coherence_doc)
    
    nTopics = len(words_per_topic)
        #number of top words per topic
    nDocs = len(coherence_corpus) #number of documents
    epsilon = 1 #smoothing parameter
       
    
    coherence= []       
    for t in range(0,nTopics): #calculate coherence
        nTopWords = len(words_per_topic[t]) 
        coherence_per_topic = []            
        for vj in range(1, nTopWords-1): #each word in topic t
                
            Dvj = 0
            for d in range(0,nDocs): 
                if (words_per_topic[t][vj] in coherence_corpus[d]): #check how many docs contain word vj
                    Dvj += 1
                    
            for vi in range((vj+1), nTopWords): 
                Dvjvi = 0
                Dvi= 0
                for d in range(0,nDocs): 
                    
                    if (words_per_topic[t][vj] in coherence_corpus[d]) and (words_per_topic[t][vi] in coherence_corpus[d]): #check how many docs contain both word vj and vi
                        Dvjvi += 1
                    if words_per_topic[t][vi] in coherence_corpus[d]:
                        Dvi+=1
                
            
                coherence_per_topic.append(math.log(Dvjvi+epsilon/(float(Dvj)*float(Dvi)),10))
        coherence.append((2/(nTopWords*(nTopWords-1)))*np.sum(coherence_per_topic))        
    print("average topic: ", (np.sum(coherence))/nTopics)

    return (np.sum(coherence))/nTopics

def GDSMM_run(phase, team, year, doc_def): 
    '''train, create and evaluate a topic model for a given team, phase, year and document definition'''
    phase_data = readTeamData(team, phase, year)
    print("got data")
    print("number of messages: " + str(phase_data.shape))
    data1 = generalPreprocessing(phase_data)
    print("general preprocessing done")
    if doc_def == "channel":
        docs = createChannelDocs(data1)
    elif doc_def == "channelDay":
        docs = createChannelDayDocs(data1)
    elif doc_def == "team":
        docs = createTeamPhaseDocs(data1)
    elif doc_def == "message":
        docs = createMessageDocs(data1)
    id2word, texts, docs_cleaned = extendedPreprocessing(docs)
    print("extended preprocessing done")
    print("number of docs: " +str(len(docs_cleaned)))
    size_docs = []
    for i in docs_cleaned:
        size_docs.append(len(i))
    print("average: " + str(np.mean(size_docs)))

    mgp, y, convergence_data = train_STTM(100, 0.1, 0.1, 10, texts, "trial")
    topic_df = show_topics(mgp)

    num_topics = sum([1 if x > 0 else 0 for x in mgp.cluster_doc_count])
    
    coherence_scores = pd.DataFrame(columns = ['coh', 'UCI', 'UMASS'])
    coh = psu_coh(mgp, y, texts)
    print(coh)
    print("number of non empty topics: ", num_topics)
    uci = coherence_uci(mgp.cluster_word_distribution, texts)
    umass = coherence_umass(mgp.cluster_word_distribution, texts)
    coherence_scores = coherence_scores.append({'coh': coh, 'UCI': uci, "UMASS": umass}, ignore_index = True)
    return convergence_data, coherence_scores, topic_df
 
def GDSMM_run_all(doc_def): 
    '''train, create and evaluate a topic model for ALL messages, for a given document definition, writes results to an excel file'''
    all_data = readTotalData()
    print("got data")
    print("number of messages: " + str(all_data.shape))
    data1 = generalPreprocessing(all_data)
    print("general preprocessing done")
    if doc_def == "channel":
        docs = createChannelDocs(data1)
    elif doc_def == "channelDay":
        docs = createChannelDayDocs(data1)
    elif doc_def == "team":
        docs = createTeamPhaseDocs(data1)
    elif doc_def == "message":
        docs = createMessageDocs(data1)
    id2word, texts, docs_cleaned = extendedPreprocessing(docs)
    print("extended preprocessing done")
    print("number of docs: " +str(len(docs_cleaned)))
    size_docs = []
    for i in docs_cleaned:
        size_docs.append(len(i))
    print("average: " + str(np.mean(size_docs)))

    mgp, y, convergence_data = train_STTM(200, 0.1, 0.1, 20, texts, "trial")
    topic_df = show_topics(mgp)

    num_topics = sum([1 if x > 0 else 0 for x in mgp.cluster_doc_count])

    coherence_scores = pd.DataFrame(columns = ['coh', 'UCI', 'UMASS'])
    coh = psu_coh(mgp, y, texts)
    print(coh)
    print("number of non empty topics: ", num_topics)

    uci = coherence_uci(mgp.cluster_word_distribution, texts)
    umass = coherence_umass(mgp.cluster_word_distribution, texts)
    coherence_scores = coherence_scores.append({'coh': coh, 'UCI': uci, "UMASS": umass}, ignore_index = True)
    writer = pd.ExcelWriter("totaltopicmodel-0.1.xlsx", engine='xlsxwriter')
    convergence_data.to_excel(writer, sheet_name='convergence')
    coherence_scores.to_excel(writer, sheet_name='coherence')
    topic_df.to_excel(writer, sheet_name='topics')
    writer.save()
    return convergence_data, coherence_scores, topic_df

    
years = [2016, 2017, 2018, 2019]
phases_2 = ["3ideas", "sketchModel", "mockupReview", "FinalSelection", "AssemblyReview", "TechnicalReview", "FinalPresentation"]
teams = ["blue", "green", "orange", "pink", "purple", "red", "silver","yellow"]
def makeModels(years, phases_2, teams):
    '''loops over all years, phases and teams and creates and evaluates a topic model for each team-phase. Takes a few days to run. Saves results to an excel file for each model'''
    for phase in phases_2:
        print(phase)
        for year in years:
            print(year)
            for team in teams:
                print(team)
                filename = phase + str(year) + team + ".xlsx"
                try: 
                    phase_data = readTeamData(team, phase, year)
                    if len(phase_data) > 0:
                        convergence_data, coherence_scores, topic_df = GDSMM_run(phase, team, year, "message")
                        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
                        convergence_data.to_excel(writer, sheet_name='convergence')
                        coherence_scores.to_excel(writer, sheet_name='coherence')
                        topic_df.to_excel(writer, sheet_name='topics')
                        writer.save()
                    
                except mysql.connector.Error as err:
                    print("Something went wrong: {}".format(err))


GDSMM_run_all("message")

########EXTRA RESOURCES NOT USED FOR APRIL SLACK PAPER BUT COULD BE USEFUL IN THE FUTURE#######
def newSTTM(team, phase, year, doc_def,  nTopics):
    '''written using the GSDMM package which I didn't end up using https://github.com/jrmazarura/GPM '''

    phase_data = readTeamData(team, phase, year)
    print("got data")
    print("number of messages: " + str(phase_data.shape))
    data1 = generalPreprocessing(phase_data)
    print("general preprocessing done")
    if doc_def == "channel":
        docs = createChannelDocs(data1)
    elif doc_def == "channelDay":
        docs = createChannelDayDocs(data1)
    elif doc_def == "team":
        docs = createTeamPhaseDocs(data1)
    elif doc_def == "message":
        docs = createMessageDocs(data1)
    id2word, texts, docs_cleaned = extendedPreprocessing(docs)


    data_dmm = GSDMM.DMM(texts, nTopics, iters =100, nTopWords = 15) # Initialize the object.

    data_dmm.topicAssigmentInitialise() # Performs the inital document assignments and counts
    data_dmm.inference()

    psi, theta, selected_psi, selected_theta = data_dmm.worddist() # Determines and stores the psi, theta and selected_psi and selected_theta values
    
    finalAssignments = data_dmm.writeTopicAssignments() # Records the final topic assignments for the documents

    coherence_topwords = data_dmm.writeTopTopicalWords(finalAssignments) # Record the top words for each document

    score = data_dmm.coherence(coherence_topwords, len(finalAssignments)) #Calculates and stores the coherence

    print("Final number of topics found: " + str(len(finalAssignments)))

    return len(finalAssignments), score, data_dmm, texts, id2word

def runSTTM(phase, team, year, doc_def, k_range, filename): 
    '''run the STTM and collect statistics for the GDSMM (other) github repository that I didn't end up using. https://github.com/jrmazarura/GPM'''
    #get the model into the docs that we need
    phase_data = readTeamData(team, phase, year)
    print("got data")
    print("number of messages: " + str(phase_data.shape))
    data1 = generalPreprocessing(phase_data)
    print("general preprocessing done")
    if doc_def == "channel":
        docs = createChannelDocs(data1)
    elif doc_def == "channelDay":
        docs = createChannelDayDocs(data1)
    elif doc_def == "team":
        docs = createTeamPhaseDocs(data1)
    elif doc_def == "message":
        docs = createMessageDocs(data1)
    id2word, texts, docs_cleaned = extendedPreprocessing(docs)
    print("extended preprocessing done")
    print("number of docs: " +str(len(docs_cleaned)))
    size_docs = []
    for i in docs_cleaned:
        size_docs.append(len(i))
    print("words per each doc: ")
    print(size_docs)
    print("average: " + str(np.mean(size_docs)))

    model_results = pd.DataFrame(columns=['K','Num_Topics_Found', 'Coherence'])
    for k in k_range:
        num_topics, coherence_score, model, texts, id2word = newSTTM(team, phase, year, "message", k)
        model_results = model_results.append({'K': k, 'Num_Topics_Found': num_topics, 'Coherence': coherence_score}, ignore_index=True)
    model_results.to_csv('GDSMM_tuning_results' + str(filename) + '.csv', index=False)
    #plot k vs coherence and num_topics
    x1 = model_results['K']
    x2 = model_results['K']

    y1 = model_results['Coherence']
    y2 = model_results['Num_Topics_Found']

    plt.subplot(1, 2, 1)
    plt.plot(x1, y1)
    plt.title('Coherence vs K')
    plt.ylabel('Coherence (UMass)')


    plt.subplot(1, 2, 2)
    plt.plot(x2, y2)
    plt.title('topics found vs K')
    plt.ylabel('Topics found')

    plt.show()