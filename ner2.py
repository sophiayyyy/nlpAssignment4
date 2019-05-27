from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from nltk import word_tokenize


# Assignment 4: NER
# This is just to help you get going. Feel free to
# add to or modify any part of it.


def getfeats(word, o, brown):
    """ This takes the word in question and
    the offset with respect to the instance
    word """
    import nltk
    posword = [word]
    pos = nltk.pos_tag(posword)
    shape = getshape(word)
    tag = '00000'
    freq = '0'
    for line in brown: #0:tag, 1:word, 2:freq
        if word == line[1]:
            tag = line[0]
            freq = line[2]

    o = str(o)
    newfeatlist = [
        (o + 'wordposbrown', word + pos[0][1]),
        (o + 'shape', shape),
        (o + 'wordbrown', word + tag + freq)
        # TODO: add more features here.
    ]
    #print(newfeatlist)
    return newfeatlist

def getshape(word):
    special = False
    hasupper = False
    if word.find('-') != -1:
        special = True
        word = word.replace("-", "")
    if word.find('_') != -1:
        special = True
        word = word.replace("-", "")
    allupper = word.isupper()
    if allupper == False:
        for i in range(len(word)):
            if check_upper(word[i]):
                hasupper = True
    if allupper and special:
            feature = '1'#'XX-XX'
    elif allupper and special == False:
        feature = '2'#'XX'
    elif hasupper and special:
        feature = '3'#'Xxx-xx'
    elif hasupper and special == False:
        feature = '4'#'Xxx'
    else:
        feature = '5'#'xx'
    return feature

def check_upper(c):
    if c >= 'A' and c <= 'Z':
        return True
    else:
        return False

def word2features(sent, i, brown):
    """ The function generates all features
    for the word at position i in the
    sentence.
    w0 + s−1, s0, s1, s−1&s0, s0&s1, s−1&s0&s1"""
    features = []
    # the window around the token
    for o in [-1,0,1]:
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            featlist = getfeats(word, o, brown)
            features.extend(featlist)
#     print(newfeatlist)
#     features.extend(newfeatlist)
    return dict(features)


if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))

    train_feats = []
    train_labels = []

    brown = []
    with open('paths') as f:
        tmp = [line.strip() for line in f]
    for line in tmp:
        tag, word, freq = line.split('\t')
        brown.append([tag, word, freq])


    for sent in train_sents:
        print("sent is:", sent)
        print("len(sent) is ", len(sent))
        for i in range(len(sent)):  # for every word in this sentences will return 3 features
            feats = word2features(sent, i, brown)
            train_feats.append(feats)
            train_labels.append(sent[i][-1])

    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(train_feats)

    # TODO: play with other models
    #model = Perceptron(verbose=1)
    model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    model.fit(X_train, train_labels)

    test_feats = []
    test_labels = []

    # # switch to test_sents for your final results
    # for sent in dev_sents:
    #     for i in range(len(sent)):
    #         feats = word2features(sent, i, brown)
    #         test_feats.append(feats)
    #         test_labels.append(sent[i][-1])
    #
    # X_test = vectorizer.transform(test_feats)
    # y_pred = model.predict(X_test)

    # switch to test_sents for your final results
    for sent in test_sents:
        for i in range(len(sent)):
            feats = word2features(sent, i, brown)
            test_feats.append(feats)
            test_labels.append(sent[i][-1])

    X_test = vectorizer.transform(test_feats)
    y_pred = model.predict(X_test)

    j = 0
    print("Writing to results_test2.txt")
    # format is: word gold pred
    with open("results_test2.txt", "w") as out:
        for sent in test_sents:
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word, gold, pred))
        out.write("\n")

    print("Now run: python conlleval.py results_test2.txt")