import nltk
import sys
import os
import string
import numpy as np

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    file_dict = {}

    # make sure it is platform independent
    data_dir = directory.replace('/', os.sep)
    data_dir = directory.replace('\\', os.sep)

    # get all files in directory
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        # get each txt file
        f = open(file_path, 'r')
        file_dict[file] = f.read()
    return file_dict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # split sentence into tokens
    tokens = nltk.word_tokenize(document)
    # make all words lowercase
    tokens_lowercase = [string.lower() for string in tokens]
    # return only proper words
    filtered_tokens = []
    for token in tokens_lowercase:
        # remove punctuation
        if token not in string.punctuation:
            # remove stopwords
            if token not in nltk.corpus.stopwords.words("english"):
                filtered_tokens.append(token)
    return filtered_tokens


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # helper func to count num documents a word is in
    def num_documents_in(word):
        count = 0
        # for every document, check the word is in it
        for document in documents.values():
            if word in set(document):
                count += 1
        return count
    # create dictionary with words and their idf's
    word_idf_dict = {}
    num_docs = len(documents.keys())
    # for every doc
    for document in documents.values():
        # for every word in that doc
        for word in document:
            # if that word's IDF hasn't been calculated
            if word not in word_idf_dict.keys():
                # calculate IDF
                num = num_documents_in(word)
                idf = np.log(num_docs / num)
                word_idf_dict[word] = idf
    return word_idf_dict


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # counts how many times a word appears
    def count_word(word, doc):
        count = 0
        for item in doc:
            if word == item:
                count += 1
        return count

    score_dict = {}
    # iterate through every file
    for file in files.keys():
        # instantiate the key/value pair
        score_dict[file] = 0
        # iterate through every word in the query
        for word in query:
            # if the word is in this file's document
            if word in set(files[file]):
                # find number of appearances
                count = count_word(word, files[file])
                # add tf-idf value to that file's score
                score_dict[file] += count * idfs[word]

    top_files = []
    # iterate for the number of desired top files
    for _ in range(n):
        keys = []
        values = []
        # add each key/value pair to lists
        for key, value in score_dict.items():
            keys.append(key)
            values.append(value)
        # return key value based off index of max value
        highest_key = keys[values.index(max(values))]
        top_files.append(highest_key)
        # avoid duplicates
        del score_dict[highest_key]
    return top_files


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # tracks sum of IDF's
    score_dict = {}
    # tracks how many query words occur (breaks ties)
    count_dict = {}
    # iterate through every file
    for sentence in sentences.keys():
        # instantiate the key/value pair
        score_dict[sentence] = 0
        count_dict[sentence] = 0
        # iterate through every word in the query
        for word in query:
            # if the word is in the doc
            if word in set(sentences[sentence]):
                # add IDF value to that file's score
                score_dict[sentence] += idfs[word]
                count_dict[sentence] += 1

    # score_dict has idf summed values of all word matches
    top_sentences = []
    # iterate for the number of desired top files
    for _ in range(n):
        keys = []
        values = []
        # add each key/value pair to lists
        for key, value in score_dict.items():
            keys.append(key)
            values.append(value)
        # find max (or ties of the max)
        max = -1
        max_val_indexes = []
        # track index for max value
        for index, val in enumerate(values):
            if val == max:
                max_val_indexes.append(index)
            elif val > max:
                max_val_indexes = [index]
                max = val
        # if it's a tie
        if len(max_val_indexes) > 1:
            proportion_dict = {}
            # create a dict that has query term densities
            for index in max_val_indexes:
                key = keys[index]
                prop = count_dict[key] / len(sentences[key])
                proportion_dict[key] = prop
            # proportion_dict has query term densities of tied keys
            max_val = np.max(list(proportion_dict.values()))
            highest_ranked_sentence = "Error"
            # find highest query term density
            for sent in proportion_dict.keys():
                if proportion_dict[sent] == max_val:
                    highest_ranked_sentence = sent
        else:
            highest_ranked_sentence = keys[max_val_indexes[0]]
        # add highest ranked sentence (key)
        top_sentences.append(highest_ranked_sentence)
        # delete it so no duplicates occur
        del score_dict[highest_ranked_sentence]
    return top_sentences


if __name__ == "__main__":
    main()
