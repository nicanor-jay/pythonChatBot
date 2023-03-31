import csv
import math
import random
import nltk
import os
import pickle
from nltk import SnowballStemmer
from nltk.corpus import stopwords, wordnet
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, multilabel_confusion_matrix
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')

                                        ###SETTING THE STEMMER###
sb_stemmer = SnowballStemmer("english")
analyzer = CountVectorizer().build_analyzer()
def stemmed_words_sb(doc):
    return (sb_stemmer.stem(w) for w in analyzer(doc))

                                        ###TRAINING CLASSIFIER###
#Load classifier, count vectorizer and tfidf transformer if they exist
#Otherwise, create them---
try:
    clf = pickle.load(open("pickle/classifier_objects/classifier.pickle", "rb"))
    count_vect_classifier = pickle.load(open("pickle/classifier_objects/count_vect_classifier.pickle", "rb"))
    tfidf_transformer_classifier = pickle.load(open("pickle/classifier_objects/tfidf_transformer_classifier.pickle", "rb"))
    # print("###Classifier exists###")
except:
    # print("###Classifier does not exist, create classifier###")

    #Setting labels
    label_dir = {
    "nameSet": "data/classifierData/nameSet",
    "nameGet": "data/classifierData/nameGet",
    "question": "data/classifierData/question",
    "talk": "data/classifierData/talk"
    }

    data = []
    labels = []

    for label in label_dir.keys():
        for file in os.listdir(label_dir[label]):
            filepath = label_dir[label] + os.sep + file
            with open(filepath, encoding='utf8', errors='ignore', mode='r') as classifierData:
                for line in classifierData:
                    content = line
                    data.append(content)
                    labels.append(label)

    #X_train ref to data of training set. y_train ref to labels of training set.
    #Similarly for X_test and y_test
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, stratify=labels, test_size=0.10, random_state=1)

    #Preparing the Bag-of-Words and training the algorithm
    count_vect_classifier = CountVectorizer(stop_words=stopwords.words('english'), analyzer=stemmed_words_sb)
    X_train_counts = count_vect_classifier.fit_transform(X_train)

    tfidf_transformer_classifier = TfidfTransformer(use_idf=True, sublinear_tf=True).fit(X_train_counts)
    X_train_tf = tfidf_transformer_classifier.transform(X_train_counts)

    #Set classifier
    clf = LogisticRegression(random_state=0).fit(X_train_tf, y_train)

    # ###TESTING CLASSIFIER###
    X_new_counts = count_vect_classifier.transform(X_test)
    X_new_tfidf = tfidf_transformer_classifier.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)

    # #Show confustion matrix for each label
    # print("Confusion Matrix: ")
    # print(str(multilabel_confusion_matrix(y_test, predicted)))
    # #Right predictions / all predictions
    # print("Accuracy Score: " + str(accuracy_score(y_test, predicted)))
    # #Weighted as it is multilabel
    # print("f1 score (question): " + str(f1_score(y_test, predicted, labels= ['question'], average='weighted')))
    # print("f1 score(small talk): " + str(f1_score(y_test, predicted, labels= ['talk'], average='weighted')))
    # print("f1 score(nameSet): " + str(f1_score(y_test, predicted, labels= ['nameSet'], average='weighted')))
    # print("f1 score(nameGet): " + str(f1_score(y_test, predicted, labels= ['nameGet'], average='weighted')))


    #Save using pickle
    with open("pickle/classifier_objects/classifier.pickle", "wb") as f:
        pickle.dump(clf, f)
    with open ("pickle/classifier_objects/count_vect_classifier.pickle", "wb") as f:
        pickle.dump(count_vect_classifier,f)
    with open("pickle/classifier_objects/tfidf_transformer_classifier.pickle", "wb") as f:
        pickle.dump(tfidf_transformer_classifier, f)

                                        ###PROCESS DATA CSV###
try:
    #Load objects using pickle if they already exist
    answers = pickle.load(open("pickle/data_objects/answers.pickle", "rb"))
    questions = pickle.load(open("pickle/data_objects/questions.pickle", "rb"))
    sources = pickle.load(open("pickle/data_objects/sources.pickle", "rb"))
    count_vect_questions = pickle.load(open("pickle/data_objects/count_vect_questions.pickle", "rb"))
    tf_transformer_questions = pickle.load(open("pickle/data_objects/tf_transformer_questions.pickle", "rb"))
    X_train_tf_questions = pickle.load(open("pickle/data_objects/X_train_tf_questions.pickle", "rb"))

    # print("###Data objects exists###")
except:
    # print("###Data objects do not exist###")
    # Get data from data.csv and put them into respective dictionaries
    answers = {}
    questions = {}
    sources = {}

    with open('data\csv\data.csv', encoding='utf8', errors='ignore', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for line in csv_reader:
            #Set current doc ID
            question_id = line[0]
            questions[question_id] = line[1]
            answers[question_id] = line[2]
            sources[question_id] = line[3]

    all_questions = questions.values()

    #process text in documents to create BoW
    count_vect_questions = CountVectorizer(stop_words=stopwords.words('english'),
                                           analyzer=stemmed_words_sb,
                                           max_df=0.11)

    X_train_counts = count_vect_questions.fit_transform(all_questions)
    tf_transformer_questions = TfidfTransformer(use_idf=True, sublinear_tf=True).\
        fit(X_train_counts)
    X_train_tf_questions = tf_transformer_questions.transform(X_train_counts)

    #Save objects using pickle
    with open("pickle/data_objects/answers.pickle", "wb") as f:
        pickle.dump(answers, f)
    with open("pickle/data_objects/questions.pickle", "wb") as f:
        pickle.dump(questions, f)
    with open("pickle/data_objects/sources.pickle", "wb") as f:
        pickle.dump(sources, f)
    with open("pickle/data_objects/count_vect_questions.pickle", "wb") as f:
        pickle.dump(count_vect_questions, f)
    with open("pickle/data_objects/tf_transformer_questions.pickle", "wb") as f:
        pickle.dump(tf_transformer_questions, f)
    with open("pickle/data_objects/X_train_tf_questions.pickle", "wb") as f:
        pickle.dump(X_train_tf_questions, f)

                                        ###PROCESS SMALL TALK CSV###
try:
    # Load objects using pickle if they already exist
    replies = pickle.load(open("pickle/talk_objects/replies.pickle", "rb"))
    questions_talk = pickle.load(open("pickle/talk_objects/questions_talk.pickle", "rb"))
    count_vect_talk = pickle.load(open("pickle/talk_objects/count_vect_talk.pickle", "rb"))
    tf_transformer_talk = pickle.load(open("pickle/talk_objects/tf_transformer_talk.pickle", "rb"))
    X_train_tf_talk = pickle.load(open("pickle/talk_objects/X_train_tf_talk.pickle", "rb"))
    # print("###Talk objects exist###")

except:
    # print("###Talk objects do not exist###")
    replies = {}
    questions_talk = {}

    with open('data\csv\\talk.csv', encoding='utf8', errors='ignore', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for line in csv_reader:
            # Set current doc ID
            question_id = line[0]
            questions_talk[question_id] = line[1]
            replies[question_id] = line[2]
    all_talk_questions = questions_talk.values()

    #process text in talk
    count_vect_talk = CountVectorizer(stop_words=stopwords.words('english'), analyzer=stemmed_words_sb)

    X_train_counts_talk = count_vect_talk.fit_transform(all_talk_questions)
    tf_transformer_talk = TfidfTransformer(use_idf=True, sublinear_tf=True).\
        fit(X_train_counts_talk)
    X_train_tf_talk = tf_transformer_talk.transform(X_train_counts_talk)

    #Save objects using pickle
    with open("pickle/talk_objects/replies.pickle", "wb") as f:
        pickle.dump(replies, f)
    with open("pickle/talk_objects/questions_talk.pickle", "wb") as f:
        pickle.dump(questions_talk, f)
    with open("pickle/talk_objects/count_vect_talk.pickle", "wb") as f:
        pickle.dump(count_vect_talk, f)
    with open("pickle/talk_objects/tf_transformer_talk.pickle", "wb") as f:
        pickle.dump(tf_transformer_talk, f)
    with open("pickle/talk_objects/X_train_tf_talk.pickle", "wb") as f:
        pickle.dump(X_train_tf_talk, f)



                                        ###GET USER NAME FIRST###
print("ChatBot: Hello! What is your name?")
userName = input(">>>")
while userName == "":
    if userName == "":
        print("Please enter a valid name")
        userName = input(">>>")
print("ChatBot: Hello " + userName+ "!")

                                        ###GET USER INPUT###
#Set input loop boolean
stop = False
#Set search threshold
searchThreshold = 0.75
print("Enter your query, or STOP to exit, and press return: ")
while not stop:
    queryInput = input(">>>")
    if queryInput == "STOP":
        stop = True
    else:
        query = [queryInput]

        #Classify query using classifier clf
        processed_newdata = count_vect_classifier.transform(query)
        processed_newdata = tfidf_transformer_classifier.transform(processed_newdata)
        classifiedQuery = clf.predict(processed_newdata)

        #Process query depending on classifier outcome

                                        ###QUESTION ROUTE###
        if classifiedQuery == 'question':
            # print("###Question detected###")
            maxSimResult = 0
            docFound = []
            print('Chatbot: You asked: "' + queryInput + '".')

            #Apply count vect to query
            X_train_counts_query = count_vect_questions.transform(query)
            X_train_query_tf = tf_transformer_questions.fit_transform(X_train_counts_query)
            query_doc = X_train_query_tf

            questions = X_train_tf_questions

            #Run search
            results = {}
            counter = 1
            
            for question in questions:
                #Calculate similarity
                sim = 1 - spatial.distance.cosine(query_doc.toarray()[0], question.toarray()[0])
                #Add similarity scores for each question to results dictionary
                results["Q" + str(counter)] = sim

                #Finding the maximum similarity scores. If multiple questions have the same score, keep them all
                #in a list
                if sim > maxSimResult and sim != maxSimResult:
                    maxSimResult = sim
                    docFound = []
                    docFound.append(counter)
                elif sim == maxSimResult:
                    docFound.append(counter)
                counter+=1

            #If similairty score could not be calculated, or the max similarity does not meet
            #the threshold, return an error message
            if math.isnan(sim):
                # If the similarity score is not an number, return an error message
                print("ChatBot: I am not able to answer this question at the moment")
            elif maxSimResult < searchThreshold:
                print("ChatBot: I was not able to understand your question, please rephrase and try again")
                # print("Max similarity was " + str(maxSimResult))
            else:
                # Output the answer relating to the highest similarity score. If multiple have the same score,
                # choose a random reply
                givenAnswer = str(random.choice(docFound))
                print("ChatBot: " + answers["Q" + givenAnswer])
                print("Source: " + sources["Q" + givenAnswer])
                # print("Similarity Score: " + str(results["Q" + givenAnswer]))
                maxSimResult = 0
                docFound = []

                                        ###SMALL TALK ROUTE###
        elif classifiedQuery == 'talk':
            # print("###Talk detected###")
            maxSimResult = 0
            docFound = []

            # Apply count vect to query
            X_train_counts_query = count_vect_talk.transform(query)
            X_train_query_tf = tf_transformer_talk.fit_transform(X_train_counts_query)
            query_doc = X_train_query_tf

            questions = X_train_tf_talk

            # Run search
            results = {}
            counter = 1
            for question in questions:
                # Calculate similarity
                sim = 1 - spatial.distance.cosine(query_doc.toarray()[0], question.toarray()[0])
                # Add similarity scores for each question to results dictionary
                results["Q" + str(counter)] = sim

                # Finding the maximum similarity scores. If multiple questions have the same score, keep them all
                # in a list
                if sim > maxSimResult and sim != maxSimResult:
                    maxSimResult = sim
                    docFound = []
                    docFound.append(counter)
                elif sim == maxSimResult:
                    docFound.append(counter)
                counter += 1

            # If similarity score could not be calculated, or the max similarity does not meet
            # the threshold, return an error message
            if math.isnan(sim):
                # If the similarity score is not a number, return an error message
                print("ChatBot: I am not able to answer this question at the moment")
            elif maxSimResult < searchThreshold:
                print("ChatBot: I was not able to understand your question, please rephrase and try again")
                # print("ChatBot: Max similarity was " + str(maxSimResult))
            else:
                # Output the answer relating to the highest similarity score. If multiple have the same score,
                # choose a random reply
                givenAnswer = str(random.choice(docFound))
                print("ChatBot: " + replies["Q" + givenAnswer])
                # print("Similarity Score: " + str(results["Q" + givenAnswer]))
                maxSimResult = 0
                docFound = []

                                        ###NAME SET ROUTE###
        elif classifiedQuery == 'nameSet':
            # print("###Name set detected###")

            #Only some names pass first time, such as Caitlyn(VB) as opposed to Richard(NNP). Discerning this difference
            #is hard to do. Thus ask the user to clarify if no such name exists

            #Check if a name (VB) exists in the string, if it does set it to user name
            post = nltk.pos_tag(word_tokenize(queryInput))
            flagNameSet = False
            for tag in post:
                if tag[1] == 'VB' or tag[1] == 'NNP':
                    userName = tag[0]
                    flagNameSet = True
            #If no name (VB) exists in string, ask the user to (re)specify name
            if flagNameSet == False:
                print("ChatBot: Please enter your new name")
                userName = input(">>>")
            print("ChatBot: Your new name is " + userName + ".")

                                        ###NAME GET ROUTE###
        elif classifiedQuery == 'nameGet':
            print("###Name get detected###")
            print("ChatBot: Your name is " + userName +".")

        print("---------------------------------")