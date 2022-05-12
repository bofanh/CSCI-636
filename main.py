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
# Support Vector Machine 
from sklearn.svm import SVC
# K Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier

# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

# reports
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score


import time

# plot 
import matplotlib.pyplot as plt

# save and load model
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
sel_list = ['star_rating' , 'review_body'] # [, 'review_headline' , 'customer_id']
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
def classifiers(X_train, y_train,X_test, X, y):
    # model file names
    model_nbc= 'nbc.sav'
    model_svm= 'svm.sav'
    model_knn= 'knn.sav'
    model_gbc= 'gbc.sav'

    print("Start Training")
    # Naive Bayes Classifier
    clf_nb = MultinomialNB()
    clf_nb.fit(X_train, y_train)
    prednb = clf_nb.predict(X_test)
    pickle.dump(clf_nb, open(model_nbc, 'wb'))


    # support vector machine
    clf_svm = SVC() # classfiyer
    clf_svm.fit(X_train, y_train) 
    predsvm=clf_svm.predict(X_test)
    pickle.dump(clf_svm, open(model_svm, 'wb'))

    #KNeighborsClassifier
    clf_neigh = KNeighborsClassifier(n_neighbors=3)
    clf_neigh.fit(X, y)
    predknn=clf_neigh.predict(X_test)
    pickle.dump(clf_neigh, open(model_knn, 'wb'))

    # GradientBoostingClassifier
    clf_gsc= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    clf_gsc.fit(X_train, y_train)
    predgsc= clf_gsc.predict(X_test)
    print("Finishing Training")
    pickle.dump(clf_gsc, open(model_gbc, 'wb'))

    return prednb,predsvm,predknn,predgsc



# classifiers report
def clfreport(prednb,predsvm,predknn,predgsc,y_test):
    nbscore = accuracy_score(y_test,prednb)
    svmscore = accuracy_score(y_test,predsvm)
    knnscore = accuracy_score(y_test,predknn)
    gscscore = accuracy_score(y_test,predgsc)
    k = 0
    target_names = ['1: N review', '5: P review']
    while k <=3:
        for i in (prednb,predsvm,predknn,predgsc):
            print("\n")
            print("                 ", ["Naive Bayes Classifier","support vector machine","KNeighborsClassifier","GradientBoostingClassifier"][k])
            print("########################################################")
            print(confusion_matrix(y_test, i))
            print('\n')
            print(classification_report(y_test, i, target_names=target_names))
            if k == 0: 
                print("Naive Bayes Classifier score is: \n", round(nbscore,2))
            elif k == 1:
                print("SVM Classifier score is: \n",round(svmscore,2))
            elif k == 2:
                print("KNN Classifier score is: \n",round(knnscore,2))
            elif k == 3:
                print("Gradient Boosting Classifier score is: \n",round(gscscore,2))
            k += 1
            print("\n")
            print("########################################################")
    return nbscore,svmscore,knnscore,gscscore




# pick the classifier with the best accurency score
def pickclf(nbscore,svmscore,knnscore,gscscore):
    winner = nbscore
    k = 0
    for i in (nbscore,svmscore,knnscore,gscscore):
        if i > winner:
            winner = i
            k += 1

    b_clf = ["Naive Bayes Classifier","support vector machine","KNeighborsClassifier","GradientBoostingClassifier"][k]
    print("\n")
    print("The best scored classifier is : ",b_clf)
    return k







# get a new review body from user input and feed it in to the best trained classifier to test out the result
def runclf(k,bow_transformer):
    pred_sample = str(input("please enter a new review_body to be predicted: \n"))
    pred_sample_transformed = bow_transformer.transform([pred_sample])


    # model file names
    model_nbc= 'nbc.sav'
    model_svm= 'svm.sav'
    model_knn= 'knn.sav'
    model_gbc= 'gbc.sav'

    try:
        if k == 0:
            loaded_model = pickle.load(open(model_nbc, 'rb'))
            a = loaded_model.predict(pred_sample_transformed)[0]
        elif k == 1:
            loaded_model = pickle.load(open(model_svm, 'rb'))
            a = loaded_model.predict(pred_sample_transformed)[0]

        elif k == 2:
            loaded_model = pickle.load(open(model_knn, 'rb'))
            a = loaded_model.predict(pred_sample_transformed)[0]
            
        elif k == 3:
            loaded_model = pickle.load(open(model_gbc, 'rb'))
            a = loaded_model.predict(pred_sample_transformed)[0]       
    except:
        print("An exception occurred")

    print("\n")
    if a == 5:
        print("This review is positive")
    elif a == 1:
        print("This review is negative")








def main(path, sel_list, des=False, dthead=False, n_head= False,n=10000):
    print("start loading data")
    data = read_data_all(path) # read data
    print("finsh loading data")
    # datades = preview_data(data,des) # Preprocess data
    # datahead = preview_data(data, dthead) # Preprocess data
    data_n = select_dropna(data,sel_list,head=n_head) # selecting wanted columnname and dropna after to make a new dataset
    print("select and dropna")
    #print(data_n['star_rating'].value_counts()) # show frequency for each star_rating 
    data_51 = data_pn(data_n) # select rows where star_rating is either 1 or 5
    print("select rating 1| 5")
    print(data_51['star_rating'].value_counts()) # first random selection
    data_51 = random_sel(data_51,n) # randomly selecting train/test dataset
    print("selecting train/test dataset")
    print(data_51['star_rating'].value_counts()) # check if the quantity for both class is balanced
    print("quantity for both class now is balanced")
    x,y = reind(data_51)
    print("reindex clean dataset")
    X = x['review_body']
    print("start spliting review body")
    bow_transformer = CountVectorizer(analyzer=text_process).fit(X) # return a vector array
    print("Splitting finished and start converting body_review")
    X = bow_transformer.transform(X)
    print("convertion finished")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=46)
    print("X_train: \n", X_train.shape)
    print("y_train: \n", y_train.shape)
    print("X_test: \n", X_test.shape)
    print("y_test: \n" , y_test.shape)
    prednb,predsvm,predknn,predgsc = classifiers(X_train, y_train,X_test, X, y) # train classifiers
    nbscore,svmscore,knnscore,gscscore = clfreport(prednb,predsvm,predknn,predgsc,y_test) # print results
    k = pickclf(nbscore,svmscore,knnscore,gscscore,) # find the best score and its classifier

    plt.bar(["Naive Bayes Classifier","support vector machine","KNeighborsClassifier","GradientBoostingClassifier"],[nbscore,svmscore,knnscore,gscscore])
    plt.ylabel('scores')
    plt.suptitle('Classifier Performance')
    plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))
    # now use the best trained classifier to do the prediction
    # runclf(k,bow_transformer)
    do = "a"
    while do != "q":
        runclf(k,bow_transformer)
        do = str(input("enter q if you want to quit:"))









start_time = time.time()
main(file_path,sel_list)
