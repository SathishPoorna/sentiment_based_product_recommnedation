
# Sentiment Based Product Recommendation System
# By Sathish Poorna



# Imports

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# Warning 
import warnings
warnings.filterwarnings('ignore')
import string
import re


# nltk
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk  import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import pairwise_distances
from numpy import *

#SKLRN
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
# Sampling
from imblearn import over_sampling
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE


# In[3]:

df = pd.read_csv("sample30.csv")
pd.options.display.max_colwidth=1000


# Shape of the Data Frame 
df.shape
#info 
df.info()

# Distribution of the Numeric Column ("reviews_rating")

df.describe()

# Checking the percentage of Null values of each coulmn 

null_check=df.isnull().sum()/len(df.index)*100
null_check.round(2)

# reviews_userCity and reviews_userProvince have more than 90% null values 


df.drop(columns=['reviews_userCity','reviews_userProvince'],axis=1,inplace=True)

df.shape

df['reviews_didPurchase'].value_counts()


# only 4.78 percent of the reviews have the purchase flas set as True , this coulmn wouldn't be useful as 95% is either null or didn't purchasehence dropping the column 

df.drop(columns=['reviews_didPurchase'],axis=1,inplace=True)

df['reviews_doRecommend'].value_counts()

#83.3 % of the data for reviews_doRecommend is skewed hence dropping the column 

df.drop(columns=['reviews_doRecommend'],axis=1,inplace=True)


df.shape

## handling Missing values 


#Dropping Null rows as they are less than 1%
df.dropna(subset = ['manufacturer','reviews_title','reviews_date','reviews_username'],inplace = True)
#df.dropna(subset = ['reviews_title'],inplace = True)


null_check=df.isnull().sum()/len(df.index)*100
null_check.round(2)


df['reviews_date'] 


# Converting reviews_date column from object type to Date Time 
df['reviews_date'] = df.reviews_date.apply(lambda x:x[0:4])


df['reviews_date'].value_counts()


# removing the 8 rows with invalid time 

df = df[df.reviews_date != ' hoo']


df.shape

## PLOTS


# we will check the rating distribution
sns.countplot(data=df,x='reviews_date')
# 2014 has the highest number of reviews 

sns.countplot(data=df, x='reviews_rating')
# Majority of the products have been rated 5

df['user_sentiment'].value_counts()


# Plotting User sentiments against the 5 ratings 

sns.countplot(df.reviews_rating, hue =df.user_sentiment)
plt.show()



df.columns

## Text Processing 

#1) Column identified for Text Processing --> reviews_text,reviews_title,user_sentiment#
#2) Combine the 2 columns to form a String collumn
#3) Remove Special characters
#4) convert to lowercase 
#5) Remove Stop Words
#6) words of length 1 or 2 would not be useful hence removing
#7) Remove unwanted white space

## Required Text Processing Functions 

# Function to Remove Special Characters
def special_character_removal(text): 
  text = "".join([char for char in text if char not in string.punctuation])
  return text

# Function to remove stop words 
stop_words = stopwords.words('english')
def stopwords_removal(text):
    words = word_tokenize(text)
    words = [wrd for wrd in words if wrd not in stop_words]
    text = " ".join(words)
    return text

df.head(5)

# Combining reviews_text and reviews_title
df['review_title_text'] = df['reviews_title']+' '+ df['reviews_text'] 

# Converting the type of review_title_text column to String type
df['review_title_text'] = df['review_title_text'].astype('str')

#User_Sentiments Column
df['user_sentiment'].value_counts()


# Mapping Negative to 0 and Positive to 1 

df['user_sentiment']= df.user_sentiment.apply(lambda x:1 if x == 'Positive' else 0 )


df['user_sentiment'].value_counts()

# converting review_title_text to lower case

df['review_title_text'] = df.review_title_text.str.lower()

# Remving Special Character ('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~') from review_title_text using string.punctuation

df['review_title_text'] = df.review_title_text.apply(lambda x : special_character_removal(x))


df['review_title_text'].head(5)


# Removing Stopwords by using teh function defined above 
df['review_title_text'] = df.review_title_text.apply(lambda X : stopwords_removal(X))
df['review_title_text'].head(5)

# we see the use of numbers in the review , Removing Numbers from review text 

df['review_title_text'] = df.review_title_text.apply(lambda X : re.sub(r'\d+', '', X))
df['review_title_text'].head(5)

# Numbers are removed 

df['review_title_text'].head(5)


# words of length 1 or 2 would not be useful hence removing
df['review_title_text'] = df['review_title_text'].replace(r'\b[a-zA-Z]{1,2}\b','', regex=True)

#Removing unwanted white spaces
df['review_title_text'] = df['review_title_text'].replace(r'^\s+|\s+$'," ", regex=True)
df['review_title_text'].head(5)

# Word Toekenisation 

df['token'] = df['review_title_text'].apply(word_tokenize)

df.head(3)

# Function for Lemmatization 
lemmatizer = WordNetLemmatizer()
def pos_tagging(tokens):
    lemmattized_array = []
    for word, tag in pos_tag(tokens):
        tag_1 = tag[0].lower()
        tag_1 = tag_1 if tag_1 in ['a', 'r', 'n', 'v'] else None
        if not tag_1:
            lemmattized_array.append(word)
        else:
            lemmattized_array.append(lemmatizer.lemmatize(word, tag_1))
    return lemmattized_array

df['lemmatized_token'] = df.token.apply(lambda x: pos_tagging(x))

df[['token','lemmatized_token']]

df['review_final'] = df['lemmatized_token'].apply(lambda x: ' '.join(word for word in x))
df['review_final'].head(5)

x=df['review_final'] 
y=df['user_sentiment']

# Splitting the dataset in to test and train
seed=100
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)

# TFIDF
word_vectorizer = TfidfVectorizer(strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 3),stop_words='english',sublinear_tf=True)
#Shape of X_Train Y_Train 
print('X_train', X_train.shape)
print('y_train', y_train.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)

# Function to display 
def display_score(classifier):
    cm = confusion_matrix(y_test, classifier.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot() 
    print(classifier)
    print('Accuracy is ', accuracy_score(y_test, classifier.predict(X_test)))
    print('Sensitivity is {}'.format(cm[1][1]/sum(cm[1])))
    print('Specificity is {}'.format(cm[0][0]/sum(cm[0])))


# Function for plotting confusion matrix

def cm_plot(cm_train,cm_test):
    print("Confusion matrix ")
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    sns.heatmap(cm_train/np.sum(cm_train), annot=True , fmt = ' .2%',cmap="Greens")
    plt.subplot(1,2,2)
    sns.heatmap(cm_test/np.sum(cm_test), annot=True , fmt = ' .2%',cmap="Greens")
    plt.show()

#Function for calculating Sensitivity and Specificity
def spec_sensitivity(cm_train,cm_test):
    #Train
    tn, fp, fn, tp = cm_train.ravel()
    specificity_train = tn / (tn+fp)
    sensitivity_train = tp / float(fn + tp)
    
    print("sensitivity for train set: ",sensitivity_train)
    print("specificity for train set: ",specificity_train)
    print("\n****\n")
    
    #Test
    tn, fp, fn, tp = cm_test.ravel()
    specificity_test = tn / (tn+fp)
    sensitivity_test = tp / float(fn + tp)
    
    print("sensitivity for test set: ",sensitivity_test)
    print("specificity for train set: ",specificity_test)

# fit_transform X_train
X_train_tf = word_vectorizer.fit_transform(X_train)
# transform X_test
X_test_tf = word_vectorizer.transform(X_test)

# Performing Over Sampling to correct the biased data 

sampling = over_sampling.RandomOverSampler(random_state=0)
X_train, y_train = sampling.fit_resample(pd.DataFrame(X_train), pd.Series(y_train))
X_train = pd.DataFrame(X_train).iloc[:,0].tolist()

X_train_transformed = word_vectorizer.transform(X_train)
X_test_transformed = word_vectorizer.transform(X_test.tolist())


## Model Building

## 1) Logistinc Regression is teh selected model


md_log = LogisticRegression()
md_log.fit(X_train_transformed,y_train)

y_train_pred_logit = md_log.predict(X_train_transformed)

print("Logistic Regression train accuracy ", accuracy_score(y_train_pred_logit, y_train),"\n")
print(classification_report(y_train_pred_logit, y_train))


# F1 score is 78%

cm_train = metrics.confusion_matrix(y_train, y_train_pred_logit)
cm_test = metrics.confusion_matrix(y_test, y_test_pred_logit)

spec_sensitivity(cm_train,cm_test)

cm_plot(cm_train,cm_test)


#Recommendation System

# Columns identified for recommendation system
#1)reviews_username
#2)name
#3)reviews_rating

recommendation_df= df[['name','reviews_username','reviews_rating']]

recommendation_df.head(5)

# Splitting the Data frame in to train and Test

train, test = train_test_split(recommendation_df, test_size=0.30, random_state=31)

train.shape, test.shape

# Unique values of user name and Products
print(train["reviews_username"].nunique())
print(train["name"].nunique())

# Creating User Product Rating Matrix with user as index , products as columns
df_pivot = train.pivot_table(index = 'reviews_username', columns = 'name', values = 'reviews_rating')
df_pivot = df_pivot.fillna(0)
df_pivot.head()


df_pivot.shape

# Dummy Train data to rate products that are not reviewed by the customer 
dummy_train = train.copy()
# Not rated products is set to 1 and rated products are set to 0
dummy_train["reviews_rating"]= dummy_train["reviews_rating"].apply(lambda x:1 if x==0 else 0)



# creating matrix with products as columns and users as rows 
df_dummy_pivot = dummy_train.pivot_table(index = 'reviews_username', columns = 'name', values = 'reviews_rating')
df_dummy_pivot = df_dummy_pivot.fillna(1)
df_dummy_pivot.head()


# User Similarity Matrix
# Create a user-Product matrix.
df_pivot = train.pivot_table(index='reviews_username',columns='name',values='reviews_rating')


mean = np.nanmean(df_pivot, axis=1)
df_sub = (df_pivot.T-mean).T

df_sub.head()

# User Similarity matrix using Pair wise dist

user_correlation = 1 - pairwise_distances(df_sub.fillna(0), metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation)
np.shape(user_correlation)

## Prediction of User based model--> ignoring negative correlated users

user_correlation[user_correlation<0]=0
user_correlation


user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
user_predicted_ratings
user_predicted_ratings.shape


## Multiplying predicted_ratings with dummy_train so that all the products that were already rated are set to 0

final_rating_user = np.multiply(user_predicted_ratings,df_dummy_pivot)



final_rating_user.head()


## Evaluation of User Based Model

# Matching users from test and train data set

match = test[test.reviews_username.isin(train.reviews_username)]
match.shape


# Creating User Product matric for evaluation 
matching_user_mat = match.pivot_table(index='reviews_username', columns='name', values='reviews_rating')


# Converting the user_correlation matrix into dataframe.
user_correlation_df = pd.DataFrame(user_correlation)
user_correlation_df['reviews_username'] = df_sub.index
user_correlation_df.set_index('reviews_username',inplace=True)
user_correlation_df.head()


username_list = match.reviews_username.tolist()


user_correlation_df.columns = df_sub.index.tolist()
user_correlation_df_1 =  user_correlation_df[user_correlation_df.index.isin(username_list)]
user_correlation_df_1.shape

user_correlation_df_2 = user_correlation_df_1.T[user_correlation_df_1.T.index.isin(username_list)]

user_correlation_df_3 = user_correlation_df_2.T


user_correlation_df_3[user_correlation_df_3<0]=0

matching_user_predicted_ratings = np.dot(user_correlation_df_3, matching_user_mat.fillna(0))
matching_user_predicted_ratings


dummy_test = match.copy()

dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)

dummy_test = dummy_test.pivot_table(index='reviews_username', columns='name', values='reviews_rating').fillna(0)
dummy_test.shape


matching_user_predicted= np.multiply(matching_user_predicted_ratings,dummy_test)



# RMSE Calculation

X = matching_user_predicted.copy() 
X = X[X>0]
scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))
print(y)


match_1 = match.pivot_table(index='reviews_username', columns='name', values='reviews_rating')


# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))

rmse = (sum(sum((match_1 - y )**2))/total_non_nan)**0.5
print(rmse)


# save the Pickle files
import pickle
pickle.dump(final_rating_user,open('user_predicted_ratings.pkl','wb'))
user_predicted_ratings =  pickle.load(open('user_predicted_ratings.pkl', 'rb'))

## User Input for Name 


user_input = input("Enter user_name")
print(user_input)

# Recommending the Top 5 products to the user.
d = user_predicted_ratings.loc[user_input].sort_values(ascending=False)[0:20]
d


# save the respective files and models through Pickle 
import pickle
pickle.dump(md_log,open('logistic_model.pkl', 'wb'))
# loading pickle object
md_log =  pickle.load(open('logistic_model.pkl', 'rb'))

pickle.dump(word_vectorizer,open('word_vectorizer.pkl','wb'))
# loading pickle object
word_vectorizer = pickle.load(open('word_vectorizer.pkl','rb'))

#Function to Filter Top 5 Products

def top_5_recommendation(user_input):
    arr = final_rating_user.loc[user_input].sort_values(ascending=False)[0:20]

    # Based on positive sentiment percentage.
    i= 0
    a = {}
    for prod_name in arr.index.tolist():
        product = prod_name
        product_name_review_list =df[df['name']== product]['review_final'].tolist()
        features= word_vectorizer.transform(product_name_review_list)
        md_log.predict(features)
        a[product] = md_log.predict(features).mean()*100
    b= pd.Series(a).sort_values(ascending = False).head(5).index.tolist()
    print("Enter Username : ",user_input)
    print("Five Recommendations for you :")
    for i,val in enumerate(b):
        print(i+1,val)

top_5_recommendation(user_input)

df.to_csv("clean_data.csv",index=False)




