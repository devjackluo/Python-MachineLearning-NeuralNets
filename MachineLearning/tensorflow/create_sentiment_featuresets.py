# [chair, table, spoon, television]
# I pulled the chair up to the table
# np.zeros(len(lexicon))
# [1 1 0 0]

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000

def create_lexicon(pos,neg):

    lexicon = []
    # read both files
    for fi in [pos,neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            # for all lines in contents
            for l in contents[:hm_lines]:
                # tokenize and all to lexicon
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)

    # lemmatizer lexicon (remove ing, ed, ly, etc.) so words like run and running both are 'run'.
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    # Counter turns lemmatized lexicon to a Counter Dictionary
    # {'the': 5545, 'and':2342}
    w_counts = Counter(lexicon)


    # Create a new lexicon by filtering words that occur too much or too little
    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            l2.append(w)

    print(len(l2))
    print(l2)

    return l2

def sample_handling(sample, lexicon, classification):

    featureset = []

    with open(sample, 'r') as f:
        contents = f.readlines()
        # for each line in file
        for l in contents[:hm_lines]:
            # lemmatize the current line
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            # create a list length of lexicon (each line has one)
            features = np.zeros(len(lexicon))
            # for all words in current line
            for word in current_words:
                # if word is in lexicon, change the index value of features corresponding to the lexicon index
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1

            # add that line feature to featureset along with the supposed classification (for testing)
            features = list(features)
            featureset.append([features, classification])

    return featureset

def create_feature_sets_and_labels(pos,neg,test_size=0.1):

    # main function to use the other two
    # create the lexicon
    lexicon = create_lexicon(pos,neg)
    features = []
    # create feature sets for both the pos and negatives
    features += sample_handling('pos.txt', lexicon, [1, 0])
    features += sample_handling('neg.txt', lexicon, [0, 1])
    # shuffle them because better that way.
    random.shuffle(features)

    # convert to a np.array for tensorflow usage
    features = np.array(features)

    # get test size number.
    test_size = int(test_size*len(features))

    # get train and test data sections
    train_x = list(features[:, 0][:-test_size])
    train_y = list(features[:, 1][:-test_size])

    test_x = list(features[:, 0][-test_size:])
    test_y = list(features[:, 1][-test_size:])

    # return train and test
    return train_x, train_y, test_x, test_y


# so the code only runs if you run it rather than importing
if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)














































