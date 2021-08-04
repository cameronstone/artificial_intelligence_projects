import nltk
nltk.download('punkt')
import sys
import numpy as np

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP
NP -> N | Adj NP | Det NP | NP Conj NP | NP PP
VP -> V | Adv VP | VP PP | VP Conj VP | V NP
PP -> P NP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    # any words w/o letter are excluded
    def validate(word):
        alphabet = set(['a','b','c','d','e','f',
        'g','h','i','j','k','l','m','n','o','p',
        'q','r','s','t','u','v','w','x','y','z'])
        for i in range(0,len(word)):
            if word[i] in alphabet:
                return True
        return False
    # split sentence into tokens
    tokens = nltk.word_tokenize(sentence)
    # make all words lowercase
    tokens_lowercase = [string.lower() for string in tokens]
    # return only words with a letter
    words = []
    for token in tokens_lowercase:
        if validate(token):
            words.append(token)
    return words


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    noun_phrases = []

    # finds all stand-alone NP's (duplicates exist)
    def get_np(tree):
        # get all non-leaf trees
        subs = list(tree.subtrees(lambda t: t.height() > 2))[1:]
        # if a tree has no non-leaves
        if subs == []:
            # check if it is a Noun Phrase
            if tree.label() == 'NP':
                # if so, append noun phrase chunk
                noun_phrases.append(tree)
        else:
            # pass all subtrees through helper func
            for sub in subs:
                get_np(sub)
    
    # get all non-leaf subtrees
    for tree in list(tree.subtrees(lambda t: t.height() > 2))[1:]:
        get_np(tree)
    
    # remove duplicates
    return np.unique(noun_phrases).tolist()


if __name__ == "__main__":
    main()
