import numpy as np
import pandas as ps
import nltk
from nltk.corpus import stopwords
import string


def process_text(text):
    nop = [c for c in text if c not in string.punctuation]
    nop = ''.join(nop)

    c_words = [word for word in nop.split() if word.lower() not in stopwords.words('english')]

    return c_words


def main():

    print("~~~~~~~~~~~~~~~~~~~EMAIL~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~SPAM~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~DETECTION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    reader = ps.read_csv("/emails.csv")
    reader.head(5)
    reader.columns
    reader.drop_duplicates(inplace=True)

    reader.isnull().sum()
    reader['text'].head().apply(process_text)
    nltk.download("stopwords")

    from sklearn.feature_extraction.text import CountVectorizer
    mess_b = CountVectorizer(analyzer=process_text).fit_transform(reader['text'])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(mess_b, reader['spam'], test_size=0.20, random_state=0)

    from sklearn.naive_bayes import MultinomialNB
    classifier = MultinomialNB().fit(X_train, y_train)

    print(classifier.predict(X_train))
    print(y_train.values)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    pred = classifier.predict(X_train)
    print(classification_report(y_train, pred))
    print()
    print('Confusion Matrix: \n', confusion_matrix(y_train, pred))
    print()
    print('Accuracy: ', accuracy_score(y_train, pred))

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    pred = classifier.predict(X_test)
    print(classification_report(y_test, pred))
    print()
    print('Confusion Matrix: \n', confusion_matrix(y_test, pred))
    print()
    print('Accuracy: ', accuracy_score(y_test, pred))

    print("~~~~~~~~~~~~~~~~ THANKS FOR USING MY PROGRAM! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print()


main()

