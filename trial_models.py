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
from wordcloud import WordCloud
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from pprint import pprint
import pyLDAvis.gensim
import pickle 
import pyLDAvis
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
from TM_preprocess import *

########
#this file is used to trial different doc types (channel, channelday) using an LDA model, to investigate how interpretable the models are
######
#LDA Gensim Tutorial: https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0 
#Gensim documentation: https://radimrehurek.com/gensim/models/ldamodel.html 
#Visualization tutorial: https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/ 

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

def BuildBaseModel(num_topics, id2word, texts, corpus):
    '''builds a base model with default settings and outputs the perplexity and coherence scores'''
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

    # Compute Perplexity
    perplexity = np.exp(lda_model.log_perplexity(corpus))
    print('\nPerplexity: ', perplexity)  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='u_mass')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    return lda_model, perplexity, coherence_lda


def compute_coherence_values2(corpus, dictionary, texts, k):
    '''calcuates coherence values for a given k, a and b hyperparameter'''
    
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=100,
                                           alpha='auto',
                                           eta='auto', 
                                           per_word_topics=True)
    
    print("lda model made")
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='u_mass')
    perplexity = np.exp(lda_model.log_perplexity(corpus))

    print("coherence values computed")

    
    return coherence_model_lda.get_coherence(), perplexity

def runHyperparameterTuning2(topic_range, corpus, id2word, texts):
    '''creates a model for every k in topic_range and saves the coherence and perplexity score, remember to change the file name'''
    grid = {}
    grid['Validation_Set'] = {}
    topics_range = topic_range
    model_results = pd.DataFrame(columns=['Topics','Coherence','Perplexity'])
    if 1 == 1:
        pbar = tqdm.tqdm(total=len(topics_range))
        for k in topics_range:
            print("k is" + str(k))
            # get the coherence score for the given parameters
            cv, perp = compute_coherence_values2(corpus=corpus, dictionary=id2word, texts=texts,
                                        k=k)
            print(k,cv, perp)
            # Save the model results
            model_results = model_results.append({'Topics': k, 'Coherence': cv, 'Perplexity': perp}, ignore_index=True)
            print("results saved")
            pbar.update(1)
        model_results.to_csv('2019bluefinalFinalPresentationMessage-tuning_results.csv', index=False)
        pbar.close()
    return model_results

def printOptimalK2(model_results):
    '''graphs the cohesion scores against different k values, different alpha values and different beta values'''
  
    plt.plot(model_results['Topics'], model_results['Coherence'])
    plt.title('Optimal number of topics by coherence')
    plt.legend()
    plt.show()

    
    plt.plot(model_results['Topics'], model_results['Perplexity'])
    plt.title('Optimal number of topics by perplexity')
    plt.legend()
    plt.show()
    


def createFinalModel(corpus, dictionary, texts, k, a,b):
    '''creates the final model with chosen k, a b, that result from hyperparameter tuning'''
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=100,
                                           alpha=a,
                                           eta=b)
    # Save model to disk.
    temp_file = datapath("2019bluephase7message")
    lda_model.save(temp_file)
    return lda_model

def pyLDAviz(lda_model, corpus, id2word, num_topics):
    '''creates an html page to display the results of the final topic model, remember to change the file name, from: https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/'''
    LDAvis_data_filepath = os.path.join('/Users/sharon/Documents/SlackData/LaurensScripts/Lda/'+ "2019blueFinalPresentationmessage"+  str(num_topics))

    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds='mmds')
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, '/Users/sharon/Documents/SlackData/LaurensScripts/Lda/'+ "2019blueFinalPresentationmessage" +  str(num_topics) +'.html')
    LDAvis_prepared

def LDAmalletBasemodel(corpus, k, id2word, texts):
    '''creates a base model (standard hyperparameters) using the LDA Mallet model, which is sometimes said to be more accurate'''
    mallet_path = '/Users/sharon/Downloads/mallet-2.0.8/bin/mallet' # update this path
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=k, id2word=id2word)
    # Show Topics
    pprint(ldamallet.show_topics(formatted=False))

    # Compute Coherence Score
    coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=texts, dictionary=id2word, coherence='u_mass')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    print('\nCoherence Score for base mallet model: ', coherence_ldamallet)


    return ldamallet, coherence_ldamallet

def compute_coherence_values_mallet(dictionary, corpus, texts, limit, start=2, step=3):
    '''calculate the coherence value for k range - hyperparameter tuning'''
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    mallet_path = '/Users/sharon/Downloads/mallet-2.0.8/bin/mallet' # update this path
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, workers = 1,corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def malletOptimalK(model_list, coherence_values):
    '''graph coherence vs k values to choose the optimal K'''
    limit=30; start=5; step=1
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    #optimal model is then model_list[best index]



def format_topics_sentences(ldamodel, corpus, texts):
    '''helper function that gets the most representative sentence for each topic, from: https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/ '''
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def showTopicSentences(optimal_model, corpus, texts):
    '''writes a csv with the most representative sentence for each topic, from: https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/'''
    df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=texts)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # Show
    print(df_dominant_topic)
    df_dominant_topic.to_csv('2019blueFinalPresentationMessageTS.csv', index=False)


def bestDocperTopic(optimal_model, corpus, texts):
    '''finds the most representative document for each topic, from: https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/'''
    df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=texts)
    sent_topics_sorteddf_mallet = pd.DataFrame()

    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                                grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                                axis=0)

    # Reset Index    
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

    # Format
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

    # Show
    print(sent_topics_sorteddf_mallet.head())
    sent_topics_sorteddf_mallet.to_csv('2019blueFinalPresentationMessageDPT.csv', index=False)

def teamPhaseModel1(phase,team,year, doc_def, num_topics,topic_range):
    '''step 1 to build a LDA model for team-phase. Gathers data and runs hyperparameter tuning'''

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
    base_model, perp1, coherence1 = BuildBaseModel(num_topics, id2word, texts, docs_cleaned)
    print("base model built")
    model_results = runHyperparameterTuning2(topic_range, docs_cleaned, id2word, texts)
    print("hyperparameter tuning done")
    printOptimalK2(model_results)

    return id2word, texts, docs_cleaned

def teamPhaseModel2(id2word, texts, docs_cleaned, num_topics, a, b):
    '''second step to make a model for each team-phase. After the idea hyperparameters have been chosen, input them here for the final model to be created, scored and saved'''
    lda = createFinalModel(docs_cleaned, id2word, texts, num_topics, a,b)
    print("final model created")
    pyLDAviz(lda, docs_cleaned, id2word, num_topics)
    print("pyLDAviz done")
    showTopicSentences(lda, docs_cleaned, texts)
    bestDocperTopic(lda, docs_cleaned, texts)

def generalMalletModel(docs_cleaned, k, id2word, texts, limit, start, step):
    '''makes an LDA mallet model if the docs and dictionary are already provided. Runs hyperparameter tuning on K and chooses the model with the highest hyperparameter values, remember to change file path and name'''
    ldamallet, coherence_ldamallet = LDAmalletBasemodel(docs_cleaned, k, id2word, texts)
    print("mallet model completed")
    model_list, coherence_values = compute_coherence_values_mallet(id2word, docs_cleaned, texts, limit, start, step)
    print("hyperparameter tuning mallet done")
    malletOptimalK(model_list, coherence_values)

    optimal_model = model_list[np.argmax(coherence_values)]
    print(optimal_model)

    # Save model to disk
    temp_file = datapath("2019blue3ideasmallet")
    optimal_model.save(temp_file)
    print("model saved to disk")
    num_topics = optimal_model.num_topics
    optimal_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(optimal_model)
    pyLDAviz(optimal_model, docs_cleaned, id2word, num_topics)
    showTopicSentences(optimal_model, docs_cleaned, texts)
    bestDocperTopic(optimal_model, docs_cleaned, texts)


def testDocsize(year, team, phase, doc_def):
    '''returns the number of docs, number of messahes, average and std of docs for each year, team, phase and definition of a document. Good for double checking the size of data before running models to make sure assumptions are met'''
    phase_data = readTeamData(team, phase, year)
    if len(phase_data) > 0: 
        num_messages = len(phase_data)
        data1 = generalPreprocessing(phase_data)
        if doc_def == "channel":
            docs = createChannelDocs(data1)
        elif doc_def == "channelDay":
            docs = createChannelDayDocs(data1)
        elif doc_def == "team":
            docs = createTeamPhaseDocs(data1)
        elif doc_def == "person":
            docs = createPersonPhaseDocs(data1)
        id2word, texts, docs_cleaned = extendedPreprocessing(docs)
        num_docs = len(docs_cleaned)
        size_docs = []
        for i in docs_cleaned:
            size_docs.append(len(i))
        avg_size_docs = np.mean(size_docs)
        std_size_docs = np.std(size_docs)
    else:
        num_messages = 0
        num_docs = 0
        avg_size_docs = 0
        std_size_docs = 0
    return num_messages, num_docs, avg_size_docs, std_size_docs

