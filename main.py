# import librarys
from importlib.resources import path
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_selection import VarianceThreshold
import nltk
# nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer # return a vector array

# Naive Bayes Classifier 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
# Support Vector Machine 
from sklearn.svm import SVC
# K Neighbors Classifier

# Gradient Boosting Classifier



import pickle




# input data 

## data location path
file_path = "amazon_reviews_us_Electronics_v1_00.tsv" # amazon electronics data file path
url = "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Outdoors_v1_00.tsv.gz"

# read data from either local or url
def read_data_all(path):
    try:
        if "http" in path:
            data = pd.read_csv(path, compression='gzip', sep='\t', on_bad_lines='skip',low_memory=False) # load and read dataset from url
        else:
            data = pd.read_csv(path,  sep='\t', on_bad_lines='skip') # load and read dataset from local path
        return data
    except:
        print("Something went wrong, check if you input the right path")


# first preview in diff ways of original loaded data. selectable
def preview_data(data, des = False, dthead = False ):
    if des == True:
        print(data.describe())
    else: pass
    if dthead == True:
        print(data.head())
    else: pass

# Feature extractor
sel_list = ['star_rating' , 'review_body' , 'review_headline' , 'customer_id']
def select_dropna(data,sel_list,head = False):
    data_n = data[sel_list] # select new dataset
    data_n =data_n.dropna() # drop NALLs in rows (axis = 0)
    if head == True: # show result
        print(data_n.head())
    else: pass
    return data_n # return new dataset


# select rows where star_rating is either 1 or 5 and equalizing the amount of sample size for 5 is same as 1
def data_pn(data_n):
    # data_51 = data_n[(data_n['star_rating'] == 1) | (data_n['star_rating'] == 5)]
    data_1 = data_n[(data_n['star_rating'] == 1)]
    data_5 = data_n[(data_n['star_rating'] == 5)]
    n = len(data_1)
    data_5n = data_5.sample(n)
    data_51 = pd.concat([data_5n,data_1])
    #print("1 has rows: ",n)
    #print("51 has rows: ",len(data_51))
    # print(data_51.shape)
    return data_51

# randomly select a fixed numbers of samples from the population
def random_sel (data_n, n=10000): # by defalt n = 1000
    data_n = data_n.sample(n) # randomly select n rows of data as sample
    print(data_n.shape)
    return data_n

# reindex the dataset
def reind(data):
    y = data['star_rating']
    x = data['review_body'].reset_index()
    return x,y



# Tokenizing words
import string
from nltk.corpus import stopwords
# stop=set(stopwords.words('english'))
def text_process(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


#Lets start training the model
from sklearn.model_selection import train_test_split
#using 30% of the data for testing, this will be revised once we do not get the desired accuracy
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=46)



# classiffier
def classifing(X_train, y_train,X_test,y_test):
    # Naive Bayes Classifier
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    pred_nb = nb.predict(X_test)

    # support vector machine









def main(path, sel_list, des=False, dthead=False, n_head= False,n=10000):
    print("start loading data")
    data = read_data_all(path) # read data
    print("finsh loading data")
    # datades = preview_data(data,des) # Preprocess data
    # datahead = preview_data(data, dthead) # Preprocess data
    data_n = select_dropna(data,sel_list,head=n_head) # selecting wanted columnname and dropna after to make a new dataset
    print("finsh select and dropna")
    #print(data_n['star_rating'].value_counts()) # show frequency for each star_rating 
    data_51 = data_pn(data_n) # select rows where star_rating is either 1 or 5
    print("finsh select rating 1| 5")
    print(data_51['star_rating'].value_counts()) # first random selection
    data_51 = random_sel(data_51,n) # randomly selecting train/test dataset
    print("finsh selecting train/test dataset")
    print(data_51['star_rating'].value_counts()) # check if the quantity for both class is balanced
    print("quantity for both class is balanced")
    x,y = reind(data_51)
    print("reindex clean dataset")
    print("vectorizing reviewbody to a vector array")
    X = x['review_body']
    bow_transformer = CountVectorizer(analyzer=text_process).fit(X) # return a vector array
    print("Splitting finished")
    X = bow_transformer.transform(X)
    print("convertion finished")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=46)
    print("X_train: \n", X_train.shape)
    print("y_train: \n", y_train.shape)
    print("X_test: \n", X_test.shape)
    print("y_test: \n" , y_test.shape)



main(file_path,sel_list)

