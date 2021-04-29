import numpy as np 
import pandas as pd 
import matplotlib.pyplot as pyplot
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
from pprint import pprint
import pyLDAvis.gensim
import pickle 
import pyLDAvis
import os
nltk.download('stopwords')

'''OLD FILE. Experimenting with LDA where each document is a channel.''' 

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

#collect data from database: we're going to collect all messages from the join of messages, users, teams and channels and put it in a dataframe, but do most of the processing in python

c.execute('SELECT * FROM message m INNER JOIN user u ON m.id_user = u.slack_uid INNER JOIN team t on u.id_team = t.id INNER JOIN channel c on c.slack_cid = m.id_channel WHERE year(m.timestamp) != 2020')
df = pd.DataFrame(c.fetchall())
df.columns = c.column_names


#preprocessing the data
#we want to remove punctuation, put everything in lower case, take out @'s and <!channels> 


#take out @'s and <channels!>

df['comments_processed'] = df['comment'].map(lambda x: re.sub(r'\<(.*[^>])>',r"", x))

#remove punctuation
translator = str.maketrans({key: None for key in string.punctuation +"123456789-"})
df['comments_processed'] = df['comments_processed'].map(lambda x: x.translate(translator)) #takes out punctuation 

#lowercase
df['comments_processed'] = df['comments_processed'].map(lambda x: x.lower())

#each channel is its own document
comments_per_channel = {}
for i in df['id_channel'].unique():
    channel_comment = df[df['id_channel']==i]
    channel_comment = list(channel_comment['comments_processed'].values)
    comments_per_channel[i] = channel_comment 
    
#create this document:
documents = []
for channel in comments_per_channel.keys():
    doc = ""
    for comment in comments_per_channel[channel]:
        doc += comment
    documents.append(doc) 
print(documents)
    


#make a wordcloud just for fun

long_string = ','.join(documents)

wordcloud = WordCloud(background_color="white", max_words=300, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
image = wordcloud.to_image()
image.show()

#prepare data for LDA

stop_words = stopwords.words('english')
stop_words.extend(['yeah', 'sorry', 'class', 'think', 'need', 'would', 'get', 'like','also'])

def sentence_to_words(sentences): #converts a document into a list of lowercase tokens, ignoring tokens that are too short or too long
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc= True, ))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


#this is where we define what a document is, so I can play with this definition 

data_words = list(sentence_to_words(documents))

data_words = remove_stopwords(data_words)
print(data_words[:1][0][:30])

#check spelling of words 
d = enchant.Dict("en_US")

data_check = [[d.check(x) for x in doc] for doc in data_words]
for i in range(1, len(data_words)): 
    data_words[i] = [j for (j,v) in zip(data_words[i], data_check[i]) if v]
print(data_words[:1][0][:30])

#genism model

# Create Dictionary
id2word = corpora.Dictionary(data_words)
# Create Corpus
texts = data_words
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1][0][:30])

#train model
num_topics = 15

lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=num_topics)

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

#analyze results

# Visualize the topics

LDAvis_data_filepath = os.path.join('/Users/sharon/Documents/SlackData/LaurensScripts/Lda/'+ "channel"+ str(num_topics))

if 1 == 1:
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
# load the pre-prepared pyLDAvis data from disk
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, '/Users/sharon/Documents/SlackData/LaurensScripts/Lda/'+ "channel" + str(num_topics) +'.html')
LDAvis_prepared