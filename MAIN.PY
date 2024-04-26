import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
from wordcloud import WordCloud
import clothing_functions
#import importlib
#importlib.reload(clothing_functions)
from clothing_functions import *
data = pd.read_csv('Womens Clothing E-Commerce Reviews.csv',index_col=False)
numeric_features = data.select_dtypes(include=[np.number])
categorical_features = data.select_dtypes(include=[np.object])
data.tail(2)

#data.columns
print('Clothing ID (unique)',len(data['Clothing ID'].unique()))
print('Total size of data',data.shape)

pd.isnull(data).sum() 
#Fill the missing data by looking at the other same key words clothes 
data_clean=data.drop(columns=['Unnamed: 0'])

for row in [136, 72, 152, 184]:
    data_clean[data['Clothing ID'] == row ]=\
    data_clean[data['Clothing ID'] == row].fillna({'Division Name':'Initmates',\
    'Department Name':'Initmate','Class Name':'Legwear'})


data_clean[data['Clothing ID'] == 492]=data_clean[data['Clothing ID'] == 492].fillna({'Division Name':'Initmates',\
                                         'Department Name':'Initmate','Class Name':'Lounge'})
data_clean[data['Clothing ID'] == 772]=data_clean[data['Clothing ID'] == 772].fillna({'Division Name':'General',\
                                         'Department Name':'Tops','Class Name':'Knits'})
data_clean[data['Clothing ID'] == 665 ]=data_clean[data['Clothing ID'] == 665].fillna({'Division Name':'Initmates',\
                                         'Department Name':'Initmate','Class Name':'Intimates'})
data_clean['Title']=data['Title'].fillna('None')
data_clean['Review Text']=data['Review Text'].fillna('None')
# create a new feature Text Length
data_clean['Text Length'] = data_clean['Review Text'].str.split().apply(len)
# creat a new feature Positively Rated
data_clean['Age'].describe()
outlier=outliers(data_clean['Age'])
data_clean['Age']=data_clean['Age'].mask(outlier['lower'],data_clean['Age'].quantile(0.05)) 
data_clean['Age']=data_clean['Age'].mask(outlier['higher'],data_clean['Age'].quantile(0.95))

# numeric columns
#sb.pairplot(data[["Clothing ID","Age", "Rating","Recommended IND"]],size = 2 ,kind ='scatter',diag_kind='kde')
data_nms=data_clean[["Age", "Rating","Recommended IND",'Positive Feedback Count',\
               'Text Length']]
sns.heatmap(data_nms.corr(),annot=True)

# Rating 
g=sns.countplot(x='Rating',data=data_clean)
data_clean.groupby(['Rating']).count().Age/len(data_clean['Rating'])

data_clean.groupby(['Recommended IND']).count().Age/len(data_clean['Recommended IND'])
f, ax = plt.subplots(1, 3, figsize=(12, 4))
# age of customers 
plot_dist(data_clean,"Age",ax=ax[0])
# positive feedback count log10()

#plot_dist(data_clean,"Positive Feedback Count",ax=ax[1])
pfc=(data_clean["Positive Feedback Count"].map(lambda i: np.log(i) if i > 0 else 0))
sns.distplot(pfc, \
               color="b", label="Skewness : %.2f  Kurtosis : %.2f"\
                %((pfc).skew(), (pfc).kurt() ),bins=25,ax=ax[1],kde=False)
ax[1].legend(loc="best")
plt.tight_layout()
# text length
plot_dist(data_clean,"Text Length",ax=ax[2])

#f, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(x='Rating',y='Age',data=data_clean,hue='Recommended IND')
#sns.barplot(x='Positively Rated',y='Positive Feedback Count',data=data_clean,ax=ax[1])

r1_recom=data_clean[(data_clean['Rating'] == 1) & (data_clean['Recommended IND'] == 1)]
r5_no_recom=data_clean[(data_clean['Rating'] == 5) & (data_clean['Recommended IND'] == 0)]
import scipy.stats as stats
data_clean['Age'].groupby(data_clean['Recommended IND']).describe()
stats.levene(data_clean['Age'][data_clean['Recommended IND'] == 1], 
               data_clean['Age'][data_clean['Recommended IND'] == 0])
stats.f_oneway(data_clean['Age'][data_clean['Recommended IND'] == 1], 
               data_clean['Age'][data_clean['Recommended IND'] == 0])
import statsmodels.api as sm
from statsmodels.formula.api import ols
results = ols('Age ~ C(Rating)', data=data[['Age','Rating']]).fit()
results.summary()
print(data_clean['Division Name'].unique())
print(data_clean['Department Name'].unique())
print(data_clean['Class Name'].unique())

f, ax = plt.subplots(1, 2, figsize=(10, 5))
sns.countplot(x='Division Name',data=data_clean,ax=ax[0],order=data_clean['Division Name'].value_counts().index)
sns.countplot(x='Department Name',data=data_clean,ax=ax[1],order=data_clean['Department Name'].value_counts().index)

plt.subplots(figsize=(8, 5))
sns.countplot(y='Class Name', data=data_clean,order=data_clean['Class Name'].value_counts().index)
ID_rating_count=data_clean.groupby(['Clothing ID']).agg(['mean','count'])['Positive Feedback Count']
num = 20
#f, ax = plt.subplots(1, 1, figsize=(8, 5))
sns.countplot(y=data_clean['Clothing ID'], \
              order=data_clean['Clothing ID'].value_counts().iloc[:num].index)
f, ax = plt.subplots(1, 2, figsize=(15, 6))
fsize = 13
sns.heatmap(pd.crosstab(data_clean['Department Name'], data_clean['Division Name']),
            annot=True, linewidths=.5, ax=ax[0], fmt='g', cmap='Oranges',
            cbar_kws={'label': 'Count'})

sns.heatmap(pd.crosstab(data_clean['Class Name'], data_clean['Department Name']),
            annot=True, linewidths=.5, ax=ax[1], fmt='g', cmap='Oranges',
            cbar_kws={'label': 'Count'})
#data.groupby(['Rating','Recommended IND']).mean()['Positive Feedback Count']
f, ax = plt.subplots(1, 3, figsize=(18, 7))
fsize = 13
sns.heatmap(pd.crosstab(data_clean['Division Name'],data_clean['Rating']),
            annot=True, linewidths=.5, ax=ax[0], fmt='g', cmap='Oranges',
            cbar_kws={'label': 'Count'})

sns.heatmap(pd.crosstab(data_clean['Department Name'],data_clean['Rating']),
            annot=True, linewidths=.5, ax=ax[1], fmt='g', cmap='Oranges',
            cbar_kws={'label': 'Count'})


sns.heatmap(pd.crosstab(data_clean['Class Name'],data_clean['Rating']),
            annot=True, linewidths=.5, fmt='g', cmap='Oranges',
            cbar_kws={'label': 'Count'},ax=ax[2])
#data.groupby(['Rating','Recommended IND']).mean()['Positive Feedback Count']
f, ax = plt.subplots(1, 3, figsize=(18, 7))
fsize = 13
sns.heatmap(pd.crosstab(data_clean['Division Name'],data_clean['Rating'],normalize = 'index'),
            annot=True, linewidths=.5, ax=ax[0], fmt='g', cmap='Oranges',
            cbar_kws={'label': 'Count'})

sns.heatmap(pd.crosstab( data_clean['Department Name'],data_clean['Rating'],normalize = 'index'),
            annot=True, linewidths=.5, ax=ax[1], fmt='g', cmap='Oranges',
            cbar_kws={'label': 'Count'})


sns.heatmap(pd.crosstab(data_clean['Class Name'],data_clean['Rating'],normalize = 'index'),
            annot=True, linewidths=.5, fmt='g', cmap='Oranges',
            cbar_kws={'label': 'Count'},ax=ax[2])
data_clean.groupby(['Recommended IND']).describe()['Text Length']
data_clean.groupby(['Rating']).describe()['Text Length']
data_nlp = data_clean[data_clean['Rating'] !=3] # remove neutrak ratings equal 3
data_nlp['Positively Rated'] = np.where(data_nlp['Rating'] > 3, 1,0) # if >3 positive 
pd.crosstab(data_nlp['Recommended IND'], data_nlp['Positively Rated'])
words_dist_review=clothing_text(data_nlp['Review Text']).word_freq()  # review 
words_dist_title=clothing_text(data_nlp['Title']).word_freq()    # title
del words_dist_title['none'] # remove the none
plot_cloud(words_dist_title,size = (15,5))
plot_cloud(words_dist_review,size = (15,5))

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (8, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
No_Recommend = data_clean['Review Text'][data_clean['Recommended IND']==0]
Recommend = data_clean['Review Text'][data_clean['Recommended IND']==1]
clothing_text(No_Recommend).word_freq().plot(20,cumulative=False)
clothing_text(Recommend).word_freq().plot(20,cumulative=False)


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for col in ['Division Name','Department Name','Class Name']:
    data_nlp[col] = le.fit_transform(data_nlp[col])
data_nlp.columns
X=data_nlp.drop(columns=['Positively Rated'])
y=data_nlp['Positively Rated']
# split the dataset for classification test size 0.3 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=5,ngram_range=(1,3)).fit(X_train['Review Text']) # convert to matrix
X_train_vectorized = vect.transform(X_train['Review Text'])
X_test_vectorized = vect.transform(X_test['Review Text'])
print('Shape of Sparse Matrix: ', X_train_vectorized.shape)
print('Amount of Non-Zero occurences: ', X_train_vectorized.nnz)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from nltk.classify import NaiveBayesClassifier
from sklearn.metrics import confusion_matrix,classification_report


LR=LogisticRegression(penalty='l1',C=2.5)
LR.fit(X_train_vectorized,y_train)
predictions = LR.predict(X_test_vectorized)

kfold = KFold(n_splits=5, shuffle=True, random_state=0)
result_LR= cross_val_score(LR, X_train_vectorized, y_train, \
                                   cv=kfold,scoring='roc_auc')
print('Cross Validation '+'Logistic Regression',np.mean(result_LR))
print('AUC:',roc_auc_score(y_test,predictions))
feature_names=np.array(vect.get_feature_names())
sorted_coef_index=LR.coef_[0].argsort()
small=list(feature_names[sorted_coef_index[:10]])
large=list(feature_names[sorted_coef_index[-11:-1]])
print(small)
print(large)
print(classification_report(y_test,predictions))

# Naive Bayes method 
nb = MultinomialNB()
nb.fit(X_train_vectorized,y_train)
predictions = nb.predict(X_test_vectorized)

result_NB= cross_val_score(nb, X_train_vectorized, y_train, \
                                   cv=kfold,scoring='roc_auc')
print('Cross Validation '+'Naive Bayes',np.mean(result_NB))
print('AUC:',roc_auc_score(y_test,predictions))

X_train_f=add_feature(X_train_vectorized, [X_train['Age'],X_train['Division Name'],\
                                           X_train['Department Name'],X_train['Class Name']])
X_test_f=add_feature(X_test_vectorized, [X_test['Age'],X_test['Division Name'],\
                                        X_test['Department Name'],X_test['Class Name']])
LR=LogisticRegression(penalty='l1',C=2.5)
LR.fit(X_train_f,y_train)
predictions = LR.predict(X_test_f)
result_LR= cross_val_score(LR, X_train_f, y_train, \
                                   cv=kfold,scoring='roc_auc')
print(result_LR.mean())
print('AUC:',roc_auc_score(y_test,predictions))

print(classification_report(y_test,predictions))
from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_train_vectorized, y_train) 
predictions = clf.predict(X_test_vectorized)
result_svm= cross_val_score(clf, X_train_vectorized, y_train, \
                                   cv=kfold,scoring='roc_auc')
print(result_svm.mean())
print('AUC:',roc_auc_score(y_test,predictions))

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('bow', TfidfVectorizer(min_df=5,ngram_range=(1,3))),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
pipeline.fit(X_train['Review Text'],y_train)
predictions = pipeline.predict(X_test['Review Text'])
print('AUC:',roc_auc_score(y_test,predictions))
X=data_clean.drop(columns=['Recommended IND'])
y=data_clean['Recommended IND']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)
vect = CountVectorizer(analyzer=text_process,min_df=5,ngram_range=(1,3)).fit(X_train['Review Text']) # convert to matrix
X_train_vectorized = vect.transform(X_train['Review Text'])
X_test_vectorized = vect.transform(X_test['Review Text'])

print('Shape of Sparse Matrix: ', X_train_vectorized.shape)
print('Amount of Non-Zero occurences: ', X_train_vectorized.nnz)


LR=LogisticRegression(penalty='l1',C=3)
LR.fit(X_train_vectorized,y_train)
predictions = LR.predict(X_test_vectorized)

kfold = KFold(n_splits=5, shuffle=True, random_state=0)
result_LR= cross_val_score(LR, X_train_vectorized, y_train, \
                                   cv=kfold,scoring='roc_auc')
print('Cross Validation '+'Logistic Regression',np.mean(result_LR))
print('AUC:',roc_auc_score(y_test,predictions))
# Naive Bayes method 
nb = MultinomialNB()
nb.fit(X_train_vectorized,y_train)
predictions = nb.predict(X_test_vectorized)
result_NB= cross_val_score(nb, X_train_vectorized, y_train, \
                                   cv=kfold,scoring='roc_auc')
print('Cross Validation '+'Naive Bayes',np.mean(result_NB))
print('AUC:',roc_auc_score(y_test,predictions))
feature_names=np.array(vect.get_feature_names())
sorted_coef_index=nb.coef_[0].argsort()
small=list(feature_names[sorted_coef_index[:20]])
large=list(feature_names[sorted_coef_index[-21:-1]])
print(small)
print(large)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
    
    