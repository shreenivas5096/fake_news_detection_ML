import pandas as pd
import sklearn
import itertools
import numpy as np
import seaborn as sb
import re
import nltk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import confusion_matrix
# from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

training_dataframe = pd.read_csv('train.csv')
training_dataframe.head(15)
training_dataframe.shape
training_dataframe['Labels'] = np.where(training_dataframe['Label']== True , 1, 0)
training_dataframe[0:10]
training_dataframe = training_dataframe.drop("Label", axis = 1)
training_dataframe.head(15)

def create_distribution(dataFile):
    return sb.countplot(x='Labels', data=dataFile, palette='hls')
create_distribution(training_dataframe)
def data_qualityCheck():
    training_dataframe.isnull().sum()
    training_dataframe.info()  
data_qualityCheck()
training_dataframe = training_dataframe.dropna()
data_qualityCheck()
training_dataframe.shape
training_dataframe.head(10)
training_dataframe.reset_index(drop= True,inplace=True)
training_dataframe.head(10)
label_train = training_dataframe.Labels
label_train.head(10)
training_dataframe = training_dataframe.drop("Labels", axis = 1)
training_dataframe.head(10)
training_dataframe['Statement'][2000]
lemmatizer = WordNetLemmatizer()
stpwrds = list(stopwords.words('english'))
stpwrds
for x in range(len(training_dataframe)) :
    corpus = []
    news_statement = training_dataframe['Statement'][x]
    news_statement = re.sub(r'[^a-zA-Z\s]', '', news_statement)
    news_statement = news_statement.lower()
    news_statement = nltk.word_tokenize(news_statement)
    for y in news_statement :
        if y not in stpwrds :
            corpus.append(lemmatizer.lemmatize(y))
    news_statement = ' '.join(corpus)
    training_dataframe['Statement'][x] = news_statement      
training_dataframe['Statement'][2000]
X_train, X_test, Y_train, Y_test = train_test_split(training_dataframe['Statement'], label_train, test_size=0.2, random_state=1)
X_train
X_train.shape
Y_train
tfidf_vector = TfidfVectorizer()
vector_X_train = tfidf_vector.fit_transform(X_train)
vector_X_test = tfidf_vector.transform(X_test)
vector_X_train.shape

loaded_model_1 = pickle.load(open('./model.pkl', 'rb'))
loaded_model_2 = pickle.load(open('./model_gnb.pkl', 'rb'))
loaded_model_3 = pickle.load(open('./model_lr.pkl', 'rb'))
loaded_model_4 = pickle.load(open('./model_rf.pkl', 'rb'))
# tfidf_vector = pickle.load(open('./model_tfidf_v.pkl', 'rb'))

def news_predictor(news):
    corpus=[]
    news_statement = news
    news_statement = re.sub(r'[^a-zA-Z\s]', '', news_statement)
    news_statement = news_statement.lower()
    news_statement = nltk.word_tokenize(news_statement)
    for y in news_statement :
        if y not in stpwrds :
            corpus.append(lemmatizer.lemmatize(y))     
    input_data = [' '.join(corpus)]
    vectorized_input_data = tfidf_vector.transform(input_data) 
    prediction_1 = loaded_model_1.predict(vectorized_input_data)
    #prediction_2 = loaded_model_2.predict(vectorized_input_data.toarray())
    prediction_3 = loaded_model_3.predict(vectorized_input_data)
    prediction_4 = loaded_model_4.predict(vectorized_input_data)
    prediction = prediction_1[0] + prediction_3[0]  + prediction_4[0]  
    if prediction <= 2:
        return 0
        # print("Prediction =>  Fake News")
    return 1
        #print("Prediction =>  Real News")
