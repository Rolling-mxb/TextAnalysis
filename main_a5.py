from nltk.corpus import movie_reviews
import nltk
import random
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from nltk.corpus import sentiwordnet as swn
import time
import pandas as pd
import re
import pickle
import argparse
documents = [(movie_reviews.raw(fileid), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)
all_words = [x[0] for x in documents]
labels = [x[1] for x in documents]
lab_array = np.array(labels).reshape(len(labels), 1)

a = [x[0] for x in documents]


def NB_set1(doc = all_words, lab = lab_array):
    start_time = time.time()
    cv = CountVectorizer(max_df=0.8, min_df=7, max_features=2500, stop_words='english')
    featureset1 = cv.fit_transform(doc)
    dataset = np.concatenate((featureset1.toarray(), lab), axis=1)
    dataset = pd.DataFrame(dataset)
    X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, -1], train_size=0.9,
                                                    random_state=0)
    NB_mod1 = MultinomialNB()
    NB_mod1.fit(X_train, y_train)
    end_time = time.time()
    with open("bayes-all-words","wb") as f:
        pickle.dump(NB_mod1, f)
    print("Elapsed time: %s"%(end_time-start_time))
    print("accuracy: %s"
       % accuracy_score(y_test, NB_mod1.predict(X_test)))

def tree_set1(doc = all_words, lab = lab_array):
    start_time = time.time()
    cv = CountVectorizer(max_df=0.8, min_df=7, max_features=2500, stop_words='english')
    featureset1 = cv.fit_transform(doc)
    dataset = np.concatenate((featureset1.toarray(), lab), axis=1)
    dataset = pd.DataFrame(dataset)
    X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, -1], train_size=0.9,
                                                    random_state=0)
    tree_mod1 = DecisionTreeClassifier()
    tree_mod1.fit(X_train, y_train)
    end_time = time.time()
    with open("trees-all-words","wb") as f:
        pickle.dump(tree_mod1, f)
    print("Elapsed time: %s"%(end_time-start_time))
    print("accuracy: %s"
       % accuracy_score(y_test, tree_mod1.predict(X_test)))

# feature set 2
def NB_set2(doc = all_words, lab = lab_array):
    cv_binary = CountVectorizer(max_df=0.8, min_df=7, max_features=2500, stop_words='english', binary=True)
    start_time = time.time()
    featureset2 = cv_binary.fit_transform(doc)
    dataset2 = np.concatenate((featureset2.toarray(), lab), axis=1)
    dataset2 = pd.DataFrame(dataset2)
    X2_train, X2_test, y2_train, y2_test = train_test_split(dataset2.iloc[:, :-1], dataset2.iloc[:, -1], train_size=0.9,
                                                        random_state=0)
    NB_mod2 = MultinomialNB()
    NB_mod2.fit(X2_train, y2_train)
    end_time = time.time()
    with open("bayes-all-words_binary","wb") as f:
        pickle.dump(NB_mod2, f)
    print("Elapsed time: %s" % (end_time - start_time))
    print("accuracy: %s"
          % accuracy_score(y2_test, NB_mod2.predict(X2_test)))

def tree_set2(doc = all_words, lab = lab_array):
    cv_binary = CountVectorizer(max_df=0.8, min_df=7, max_features=2500, stop_words='english', binary=True)
    start_time = time.time()
    featureset2 = cv_binary.fit_transform(doc)
    dataset2 = np.concatenate((featureset2.toarray(), lab), axis=1)
    dataset2 = pd.DataFrame(dataset2)
    X2_train, X2_test, y2_train, y2_test = train_test_split(dataset2.iloc[:, :-1], dataset2.iloc[:, -1], train_size=0.9,
                                                        random_state=0)
    tree_mod2 = DecisionTreeClassifier()
    tree_mod2.fit(X2_train, y2_train)
    end_time = time.time()
    with open("trees-all-words_binary","wb") as f:
        pickle.dump(tree_mod2, f)
    print("Elapsed time: %s" % (end_time - start_time))
    print("accuracy: %s"
          % accuracy_score(y2_test, tree_mod2.predict(X2_test)))

# print("Naive bayes accuracy on feature set 2: %s"
#       % accuracy_score(y2_test, NB_mod2.predict(X2_test)))
# print("Decision accuracy on feature set 2: %s"
#       % accuracy_score(y2_test, tree_mod2.predict(X2_test)))

# feature set 3
senti_word = [w.split() for w in all_words]

for i in range(len(senti_word)):
    senti_word[i] = [w for w in senti_word[i] if list(swn.senti_synsets(w))]
    senti_word[i] = ' '.join([w for w in senti_word[i] if
                              list(swn.senti_synsets(w))[0].pos_score() > 0.5 or list(swn.senti_synsets(w))[
                                  0].neg_score() > 0.5])
def NB_set3(doc = senti_word, lab = lab_array):
    start_time = time.time()
    cv = CountVectorizer(max_df=0.8, min_df=7, max_features=2500, stop_words='english')
    featureset3 = cv.fit_transform(doc)
    dataset3 = np.concatenate((featureset3.toarray(), lab), axis=1)
    dataset3 = pd.DataFrame(dataset3)
    X3_train, X3_test, y3_train, y3_test = train_test_split(dataset3.iloc[:, :-1], dataset3.iloc[:, -1], train_size=0.9,
                                                        random_state=0)
    NB_mod3 = MultinomialNB()
    NB_mod3.fit(X3_train, y3_train)
    end_time = time.time()
    with open("bayes-SentiWordNet_words","wb") as f:
        pickle.dump(NB_mod3, f)
    print("Elapsed time: %s" % (end_time - start_time))
    print("accuracy: %s"
          % accuracy_score(y3_test, NB_mod3.predict(X3_test)))


def tree_set3(doc = senti_word, lab = lab_array):
    start_time = time.time()
    cv = CountVectorizer(max_df=0.8, min_df=7, max_features=2500, stop_words='english')
    featureset3 = cv.fit_transform(doc)
    dataset3 = np.concatenate((featureset3.toarray(), lab), axis=1)
    dataset3 = pd.DataFrame(dataset3)
    X3_train, X3_test, y3_train, y3_test = train_test_split(dataset3.iloc[:, :-1], dataset3.iloc[:, -1], train_size=0.9,
                                                        random_state=0)
    tree_mod3 = DecisionTreeClassifier()
    tree_mod3.fit(X3_train, y3_train)
    end_time = time.time()
    with open("trees-SentiWordNet_words","wb") as f:
        pickle.dump(tree_mod3, f)
    print("Elapsed time: %s" % (end_time - start_time))
    print("accuracy: %s"
          % accuracy_score(y3_test, tree_mod3.predict(X3_test)))

# print("Naive bayes accuracy on feature set 3: %s"
#       % accuracy_score(y3_test, NB_mod3.predict(X3_test)))
# print("Decision accuracy on feature set 3: %s"
#       % accuracy_score(y3_test, tree_mod3.predict(X3_test)))

# feature set 4
with open("data\subjectivity_clues_hltemnlp05\subjclueslen1-HLTEMNLP05.tff") as f:
    MPQA_line = [line for line in f.readlines()]
    f.close()
MPQA_words = [i for [i] in [re.findall(r"word1=(.+?)\s", w) for w in MPQA_line]]

MPQA = [w.split() for w in all_words]
for i in range(len(MPQA)):
    MPQA[i] = [w for w in MPQA[i] if w in MPQA_words]
    MPQA[i] = ' '.join([w for w in MPQA[i]])

def NB_set4(doc = MPQA, lab = lab_array):
    start_time = time.time()
    cv = CountVectorizer(max_df=0.8, min_df=7, max_features=2500, stop_words='english')
    featureset4 = cv.fit_transform(doc)
    dataset4 = np.concatenate((featureset4.toarray(), lab), axis=1)
    dataset4 = pd.DataFrame(dataset4)
    X4_train, X4_test, y4_train, y4_test = train_test_split(dataset4.iloc[:, :-1], dataset4.iloc[:, -1], train_size=0.9,
                                                        random_state=0)
    NB_mod4 = MultinomialNB()
    NB_mod4.fit(X4_train, y4_train)
    end_time = time.time()
    with open("bayes-Subjectivity_Lexicon_words","wb") as f:
        pickle.dump(NB_mod4, f)
    print("Elapsed time: %s" % (end_time - start_time))
    print("accuracy: %s"
          % accuracy_score(y4_test, NB_mod4.predict(X4_test)))


def tree_set4(doc = MPQA, lab = lab_array):
    start_time = time.time()
    cv = CountVectorizer(max_df=0.8, min_df=7, max_features=2500, stop_words='english')
    featureset4 = cv.fit_transform(doc)
    dataset4 = np.concatenate((featureset4.toarray(), lab), axis=1)
    dataset4 = pd.DataFrame(dataset4)
    X4_train, X4_test, y4_train, y4_test = train_test_split(dataset4.iloc[:, :-1], dataset4.iloc[:, -1], train_size=0.9,
                                                        random_state=0)
    NB_mod4 = MultinomialNB()
    NB_mod4.fit(X4_train, y4_train)
    tree_mod4 = DecisionTreeClassifier()
    tree_mod4.fit(X4_train, y4_train)
    end_time = time.time()
    with open("trees-Subjectivity_Lexicon_words","wb") as f:
        pickle.dump(tree_mod4, f)
    print("Elapsed time: %s" % (end_time - start_time))
    print("accuracy: %s"
          % accuracy_score(y4_test, tree_mod4.predict(X4_test)))

# feature set 1 with negation

def negate(text):
    text = nltk.word_tokenize(text)
    l = []
    punctuation = "?.,!:;"
    for i in range(len(text)):
        if text[i] in {"not", "n't", "no"}:
            l.append(i)
    for i in l:
        cur = i + 1
        while cur < len(text) and text[cur] not in punctuation:
            text[cur] = "not_" + text[cur]
            cur += 1
    return " ".join(text)


all_words_negated = [negate(w) for w in all_words]





def NB_set5(doc = all_words_negated, lab = lab_array):
    start_time = time.time()
    cv = CountVectorizer(max_df=0.8, min_df=7, max_features=2500, stop_words='english')
    featureset5 = cv.fit_transform(doc)
    dataset5 = np.concatenate((featureset5.toarray(), lab), axis=1)
    dataset5 = pd.DataFrame(dataset5)
    X5_train, X5_test, y5_train, y5_test = train_test_split(dataset5.iloc[:, :-1], dataset5.iloc[:, -1], train_size=0.9,
                                                    random_state=0)
    NB_mod5 = MultinomialNB()
    NB_mod5.fit(X5_train, y5_train)
    end_time = time.time()
    with open("bayes-all_words_Negation","wb") as f:
        pickle.dump(NB_mod5, f)
    print("Elapsed time: %s" % (end_time - start_time))
    print("accuracy: %s"
          % accuracy_score(y5_test, NB_mod5.predict(X5_test)))


def tree_set5(doc = all_words_negated, lab = lab_array):
    start_time = time.time()
    cv = CountVectorizer(max_df=0.8, min_df=7, max_features=2500, stop_words='english')
    featureset5 = cv.fit_transform(doc)
    dataset5 = np.concatenate((featureset5.toarray(), lab), axis=1)
    dataset5 = pd.DataFrame(dataset5)
    X5_train, X5_test, y5_train, y5_test = train_test_split(dataset5.iloc[:, :-1], dataset5.iloc[:, -1], train_size=0.9,
                                                    random_state=0)
    tree_mod5 = DecisionTreeClassifier()
    tree_mod5.fit(X5_train, y5_train)
    end_time = time.time()
    with open("trees-all_words_Negation","wb") as f:
        pickle.dump(tree_mod5, f)
    print("Elapsed time: %s" % (end_time - start_time))
    print("accuracy: %s"
          % accuracy_score(y5_test, tree_mod5.predict(X5_test)))

def run_func(num,text):
    if num == 1:
        cv = CountVectorizer(max_df=0.8, min_df=7, max_features=2500, stop_words='english')
        text = cv.fit_transform(text)
        text = text.toarray()
        text= pd.DataFrame(text)
        with open("bayes-all-words", 'rb') as f:
            mod = pickle.load(f)
        x = mod.predict(text)
        #print(max(x, key=x.count))
        print(x)

    if num == 2:
        cv = CountVectorizer(max_df=0.8, min_df=7, max_features=2500, stop_words='english',binary=True)
        text = cv.fit_transform(text)
        text = text.toarray()
        text = pd.DataFrame(text)
        with open("bayes-all-words_binary", 'rb') as f:
            mod = pickle.load(f)
        x = mod.predict(text)
        print(x)
    if num == 3:
        ccv = CountVectorizer(max_df=0.8, min_df=7, max_features=2500, stop_words='english')
        text = cv.fit_transform(text)
        text = text.toarray()
        text= pd.DataFrame(text)
        with open("bayes-SentiWordNet_words", 'rb') as f:
            mod = pickle.load(f)
        x = mod.predict(text)
        print(x)
    if num == 4:
        cv = CountVectorizer(max_df=0.8, min_df=7, max_features=2500, stop_words='english')
        text = cv.fit_transform(text)
        text = text.toarray()
        text = pd.DataFrame(text)
        with open("bayes-Subjectivity_Lexicon_words", 'rb') as f:
            mod = pickle.load(f)
        x = mod.predict(text)
        print(x)
    if num == 5:
        cv = CountVectorizer(max_df=0.8, min_df=7, max_features=2500, stop_words='english')
        text = cv.fit_transform(text)
        text = text.toarray()
        text = pd.DataFrame(text)
        with open("bayes-all_words_Negation", 'rb') as f:
            mod = pickle.load(f)
        x = mod.predict(text)
        print(x)



# =============================================================================

# Create an empty parser
parser = argparse.ArgumentParser()

# add --train, --run to the parser created above
parser.add_argument("--train", help="Train classifier on different feature sets",
                    action="store_true")
parser.add_argument("--run", help="Run on test text",
                    action="run")


# Write the conditions of triggering each parser
args = parser.parse_args()
if args.train:
    NB_set1(doc = all_words, lab = lab_array)
    tree_set1(doc=all_words, lab=lab_array)
    NB_set2(doc=all_words, lab=lab_array)
    tree_set2(doc=all_words, lab=lab_array)
    NB_set3(doc=senti_word, lab=lab_array)
    tree_set3(doc=senti_word, lab=lab_array)
    NB_set4(doc=MPQA, lab=lab_array)
    tree_set4(doc=MPQA, lab=lab_array)
    NB_set5(doc=all_words_negated, lab=lab_array)
    tree_set5(doc=all_words_negated, lab=lab_array)

if args.run:
    run_func(args.run)
