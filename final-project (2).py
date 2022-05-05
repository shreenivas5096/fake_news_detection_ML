#!/usr/bin/env python
# coding: utf-8

# In[50]:


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
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# In[2]:


train_df = pd.read_csv(r'C:\Users\Shreenivas\train.csv')


# In[3]:


train_df.head(15)


# In[4]:


# train_df = train_df.drop("title", axis = 1)
# train_df = train_df.drop("id", axis = 1)


# In[5]:


train_df.shape
train_df['Labels'] = np.where(train_df['Label']== True , 1, 0)
train_df[0:10]


# In[6]:


train_df = train_df.drop("Label", axis = 1)


# In[7]:


train_df.head(15)


# In[8]:


def create_distribution(dataFile):
    return sb.countplot(x='Labels', data=dataFile, palette='hls')

# by calling below we can see that training, test and valid data seems to be failry evenly distributed between the classes
create_distribution(train_df)


# In[9]:


def data_qualityCheck():
    print("Checking data qualitites...")
    train_df.isnull().sum()
    train_df.info()  
    print("check finished.")
data_qualityCheck()


# In[10]:


# Drop NULL Values
train_df = train_df.dropna()


# In[11]:


data_qualityCheck()


# In[12]:


train_df.shape


# In[13]:


train_df.head(10)


# In[14]:


train_df.reset_index(drop= True,inplace=True)


# In[15]:


train_df.head(10)


# In[16]:


label_train = train_df.Labels


# In[17]:


label_train.head(10)


# In[18]:


train_df = train_df.drop("Labels", axis = 1)


# In[19]:


train_df.head(10)


# In[20]:


train_df['Statement'][2000]


# In[21]:


lemmatizer = WordNetLemmatizer()
stpwrds = list(stopwords.words('english'))


# In[22]:


stpwrds


# In[23]:


for x in range(len(train_df)) :
    corpus = []
    review = train_df['Statement'][x]
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    review = review.lower()
    review = nltk.word_tokenize(review)
    for y in review :
        if y not in stpwrds :
            corpus.append(lemmatizer.lemmatize(y))
    review = ' '.join(corpus)
    train_df['Statement'][x] = review      


# In[24]:


train_df['Statement'][2000]


# In[25]:


X_train, X_test, Y_train, Y_test = train_test_split(train_df['Statement'], label_train, test_size=0.2, random_state=1)


# In[26]:


X_train


# In[27]:


X_train.shape


# In[28]:


Y_train


# In[29]:


tfidf_v = TfidfVectorizer()
tfidf_X_train = tfidf_v.fit_transform(X_train)
tfidf_X_test = tfidf_v.transform(X_test)


# In[30]:


tfidf_X_train.shape


# In[31]:


# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
    
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')


# In[32]:


classifier = PassiveAggressiveClassifier()
classifier.fit(tfidf_X_train,Y_train)


# In[52]:


Y_pred = classifier.predict(tfidf_X_test)
score = metrics.accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {round(score*100,2)}%')
cm = metrics.confusion_matrix(Y_test, Y_pred)
plot_confusion_matrix(classifier, tfidf_X_test, Y_test)


# In[34]:


clf = RandomForestClassifier(n_estimators = 100) 


# In[35]:


clf.fit(tfidf_X_train, Y_train)


# In[36]:


y_pred = clf.predict(tfidf_X_test)


# In[37]:


# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(Y_test, y_pred)*100)


# In[54]:


cm_RF = metrics.confusion_matrix(Y_test, y_pred)
plot_confusion_matrix(clf, tfidf_X_test, Y_test)


# In[39]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(tfidf_X_train.toarray(), Y_train)


# In[40]:


y_pred_nb = gnb.predict(tfidf_X_test.toarray())
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(Y_test, y_pred_nb)*100)


# In[57]:


cm_nb = metrics.confusion_matrix(Y_test, y_pred_nb)
plot_confusion_matrix(gnb, tfidf_X_test.toarray(), Y_test)


# In[42]:


clf_lr = LogisticRegression(random_state=0).fit(tfidf_X_train, Y_train)


# In[43]:


y_pred_lr=clf_lr.predict(tfidf_X_test)
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(Y_test, y_pred_lr)*100)


# In[58]:


cm_lr = metrics.confusion_matrix(Y_test, y_pred_lr)
plot_confusion_matrix(clf_lr, tfidf_X_test, Y_test)


# In[45]:


pickle.dump(classifier,open('./model.pkl', 'wb'))
pickle.dump(gnb,open('./model_gnb.pkl', 'wb'))
pickle.dump(clf_lr,open('./model_lr.pkl', 'wb'))
pickle.dump(clf,open('./model_rf.pkl', 'wb'))


# In[46]:


# load the model from disk
loaded_model = pickle.load(open('./model.pkl', 'rb'))


# In[47]:


def fake_news_predictor(news):
    review = news
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    review = review.lower()
    review = nltk.word_tokenize(review)
    for y in review :
        if y not in stpwrds :
            corpus.append(lemmatizer.lemmatize(y))     
    input_data = [' '.join(corpus)]
    vectorized_input_data = tfidf_v.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    if prediction[0] == 0:
        print("Prediction =>  Fake News")
    else:
        print("Prediction =>  Real News")


# In[48]:


fake_news_predictor("Obama is from India")

