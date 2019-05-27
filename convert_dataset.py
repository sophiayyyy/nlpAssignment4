from nltk.corpus import conll2002

if __name__ == "__main__":
    # Load the training data
    train_sents = list(conll2002.iob_sents('esp.train'))
    # dev_sents = list(conll2002.iob_sents('esp.testa'))
    # test_sents = list(conll2002.iob_sents('esp.testb'))
    res = []
    output = ''
    train_feats = []
    for sent in train_sents:
        output = ''
        for i in range(len(sent)):
            output = output + ' ' + sent[i][0]
        res.append(output)
        #print("output is:", output)
        print("len(sent) is ", len(sent))

    print("Writing to browncluster_train.txt")
    # format is: word gold pred
    with open("browncluster_train.txt", "w") as out:
        for sent in res:
            out.write(sent)
            out.write("\n")

    print("Now run: python conlleval.py results.txt")