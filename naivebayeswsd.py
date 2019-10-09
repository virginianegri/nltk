from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk import WordNetLemmatizer
from nltk.probability import FreqDist  # this is useful, somewhere...
import random


def extract_context(list_of_lemmas, idx, context_limit):
    '''
    Extracts the left and right contexts (i.e., lists of lemmas) of the lemma in position 'idx'
    - list_of_lemmas: list containing a lemmatized text;
        e.g.: ['jenn_barthole', 'apple', 'survey', 'are', 'apple', 'product', 'good', ...]
    - idx: the position of the target lemma inside the list of lemmas
    - context_limit: how many lemmas to consider, before and after idx
    - returns:
         - the list of lemmas, with length context_limit, before idx;
           complete with '' if no enough lemmas are available before idx
         - the list of lemmas, with length context_limit, after idx;
           complete with '' if no enough lemmas are available after idx
    '''

    left_context = []
    right_context = []
    for i in range(1, context_limit + 1):
        # get left context
        if (idx - i > 0 and list_of_lemmas[idx - i] != ""):
            left_context.append(list_of_lemmas[idx - i])
        # get right context
        if (idx + i < len(list_of_lemmas) and list_of_lemmas[idx + i] != ""):
            right_context.append(list_of_lemmas[idx + i])

    return left_context, right_context
    # E.g. ['jenn_barthole']   and   ['survey']


def get_best_co_occurring_lemmas(target_lemma, n, context_limit, texts):
    '''
    Returns the n most frequently co-occurring lemmas of the given target lemma
    - target_lemma: the lemma for which the co-occurring lemma must be found
    - n: how many co-occurring lemmas to retain
    - context_limit: how many lemmas to consider, before and after the target lemma
    - texts: the corpus; an hashtable where items are file names;
     e.g.: {'COMPANY': './data/apple-company-training.txt', 'FRUIT': './data/apple-fruit-training.txt'}
    - returns: a list of the n most co-occurring lemmas
    '''

    tokens = []
    for textPath in texts.values():
        f = open(textPath)
        for word in f.read().split():
            tokens.append(word)

    co_occurrences_dict = {}
    for j in range(len(tokens)):
        if (tokens[j] == target_lemma):
            for i in range(1, context_limit + 1):
                if ((i - j) > 0):
                    # left tokens
                    if (tokens[i - j] in co_occurrences_dict):
                        # update
                        currentOccurences = co_occurrences_dict[tokens[i - j]]
                        co_occurrences_dict[tokens[i - j]] = currentOccurences + 1
                    else:
                        # add new token
                        co_occurrences_dict[tokens[i - j]] = 1
                if ((i + j) < len(tokens)):
                    # right tokens
                    if (tokens[i + j] in co_occurrences_dict):
                        # update
                        currentOccurences = co_occurrences_dict[tokens[i + j]]
                        co_occurrences_dict[tokens[i + j]] = currentOccurences + 1
                    else:
                        # add new token
                        co_occurrences_dict[tokens[i + j]] = 1

    sorted(co_occurrences_dict.items(), key=lambda x: x[1])
    co_occurrences = list(co_occurrences_dict.keys())
    return co_occurrences[:n]  # The first n lemmas (n most frequent co-occurring lemmas
    # e.g., ['crisp', 'http', 'ipad', 'iphone', 'juice', 'like', 'mac', 'make', 'making', 'pie', 'product', 'sauce']


def features(target_word, left_context_lemmas, right_context_lemmas, best_co_occurring_lemmas):
    '''
    Returns the feature set of a given target lemma: lemma, left collocations, right collocations, co_occurrence
    - target_word: the word to classify
    - left_context_lemmas: a list of lemmas, before the target word; e.g.: ['jenn_barthole']
    - right_context_lemmas: a list of lemmas, after the target word; e.g.: ['survey']
    - best_co_occurring_lemmas: the list of best co-occurring lemmas, for the target word;
      e.g.: e.g., ['crisp', 'http', 'ipad', 'iphone', 'juice', 'like', 'mac', 'make', 'making', 'pie', 'product', 'sauce']
    - returns: an hashtable {'word': ..., 'left_collocations': ..., 'right_collocations': ..., 'co_occurrence': ...}
    '''
    co_occurence_vector = [0] * len(best_co_occurring_lemmas)
    for i in range(len(best_co_occurring_lemmas)):
        for j in range(len(left_context_lemmas)):
            if left_context_lemmas[j] == best_co_occurring_lemmas[i]:
                co_occurence_vector[i] += 1
        for j in range(len(right_context_lemmas)):
            if right_context_lemmas[j] == best_co_occurring_lemmas[i]:
                co_occurence_vector[i] += 1

    return {'word': target_word, 'left_collocations': tuple(left_context_lemmas),
            'right_collocations': tuple(right_context_lemmas), 'co_occurrence': tuple(co_occurence_vector)}
    # e.g.: {'left_collocations': ('jenn_barthole'), 'right_collocations': ('survey'),
    #     'co_occurrence': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 'word': 'apples'}


def add_lemmas(text):
    '''
	Lemmatizes the text, using WordNet; returns (lemma, word) for each word in text
	NB: discards stopwords and tokens of 1 character
	- text: list of tokens of the document to lemmatize
	- returns: a list of pairs (lemma, word)
	'''
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(text)  # notice that the tokenizer splits words like '#apple' into '#', 'apple'
    lem = WordNetLemmatizer()
    # stop words are words not carrying particular meaning (e.g. particles)
    # we also want to discard symbols such as ' -> check if len(x)>1
    result = [(lem.lemmatize(x.lower()), x.lower()) for x in tokens if
              x not in stopwords.words('english') and len(x) > 1]
    return result  # the pairs (lemma, word)


def main():
    # Define constants
    train_test_texts = {}  # a void hash table
    train_test_texts['FRUIT'] = './data/apple-fruit-training.txt'
    train_test_texts['COMPANY'] = './data/apple-company-training.txt'
    experiment_text = './data/apple-tweets.txt'
    lemma_to_classify = 'apple'  # Give it in its base form, so 'apple' and 'apples' are both classified
    context_limit = 1  # Context will be +/- 1 (context window)
    n_for_co_occurring = 12  # How many co-occurring lemmas to retain; e.g. retain the 12 best co-occurring lemmas
    train_set_fraction = 0.8  # 80 %

    # Build vector of the best co-occurring lemmas, for all the meaning of the lemma to classify
    best_co_occurring_lemmas = get_best_co_occurring_lemmas(lemma_to_classify, n_for_co_occurring,
                                                            context_limit, train_test_texts)

    # Loop through each item, grab the text, tokenize it and create a training feature with it (create corpus)
    featuresets = []
    for sense, training_file in iter(
            train_test_texts.items()):  # iter() creates couples (index, item); index is a string here
        print("Training %s..." % sense)
        text = open(training_file, 'r', encoding='utf-8').read()  # contains all the content of the file
        list_of_lemmas_words = add_lemmas(text)  # tuples (lemma, token)
        list_of_lemmas = [x[0] for x in list_of_lemmas_words]  # Retains only lemmas
        count_lemma_to_classify = 0
        for idx, (lemma, word) in enumerate(
                list_of_lemmas_words):  # After enumerate():[(1,('cat','cats')),(2,('dog','dog')),...(lemma, token)]
            if lemma == lemma_to_classify:
                left_context_lemmas, right_context_lemmas = extract_context(list_of_lemmas, idx,
                                                                            context_limit)  # extract preceding and following lemma
                # Append a new tuple to the list
                # Notice we use word, not lemma as a feature
                featuresets += [(features(word, left_context_lemmas, right_context_lemmas, best_co_occurring_lemmas),
                                 sense)]  # dictionary containing features
                count_lemma_to_classify += 1

        # print how many samples of lemma_to_classify, for the current sense
        # We need to verify that the corpus is balances (same length of two files companies-fruits) ->this guarantees accuracy
        print("For sense \"%s\" there are %d samples of \"%s\"" % (sense, count_lemma_to_classify, lemma_to_classify))

    # Select training set and test set
    # Shuffling is needed so that train_set and test_set will contain samples from both the first and the second file
    random.shuffle(featuresets)
    train_set_limit = int(train_set_fraction * len(featuresets))
    train_set, test_set = featuresets[:train_set_limit], featuresets[train_set_limit:]

    # Train...
    classifier = NaiveBayesClassifier.train(train_set)

    # Test... Notice that each run will result in a different accuracy, as the train set is randomly chosen
    print("Accuracy:", accuracy(classifier, test_set))

    # Try to classify a new file and print the surrounding words of the classified lemma
    print("\nClassify new text: %s" % experiment_text)
    text = open(experiment_text, 'r', encoding='utf-8').read()
    list_of_lemmas_words = add_lemmas(text)
    list_of_lemmas = [x[0] for x in list_of_lemmas_words]  # Retains only lemmas of the current text
    list_of_words = [x[1] for x in list_of_lemmas_words]    # Retains only tokens of the current text
    for idx, (lemma, word) in enumerate(
            list_of_lemmas_words):  # After enumerate(): [(1,('cat','cats')),(2,('dog','dog')), ...]
        if lemma == lemma_to_classify:
            left_context_lemmas, right_context_lemmas = extract_context(list_of_lemmas, idx, context_limit)
            decision = classifier.classify(
                features(word, left_context_lemmas, right_context_lemmas, best_co_occurring_lemmas))

            # - Extract at most 10 tokens before and the target lemma
            left_surrounding, right_surrounding = extract_context(list_of_words, idx, 10)
            print("Class: %s\tWhere(+-10): %s *%s* %s" % (decision, left_surrounding, lemma, right_surrounding))
        # E.g.: Class: COMPANY	Where(+-10): ['noosy', 'offers', 'hdmi', 'adapter', 'for', 'the', 'ipad', 'iphone',
        #       'ipod', 'touch'] *apple* ['ipad', 'iphone', 'http', '://', 'bit', 'ly']

    ####################################################################################################################
    # K-fold validation
    k = 20  # set k for k-fold validation
    k_size = len(featuresets) // k
    random.shuffle(featuresets)
    avg_accuracy = 0
    for i in range(0, k):
        start = i * k_size
        end = i * k_size + k_size
        valid_set = featuresets[start:end]
        train_set = []
        if i == 0:
            train_set = featuresets[end:]
        elif i == k - 1:
            train_set = featuresets[:start]
        else:
            train_set = featuresets[:start] + featuresets[end:]

        classifier = NaiveBayesClassifier.train(train_set)
        avg_accuracy += accuracy(classifier, valid_set)
    avg_accuracy = avg_accuracy / k

    print("The average accuracy is", avg_accuracy)


if __name__ == '__main__':
    main()
