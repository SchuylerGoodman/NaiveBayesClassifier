import os
import os.path
import re
import sets

class Classifier:

    """
    default class variables
    """
    dataset_dir = "20_newsgroups/"
    training_indices_dir = dataset_dir + "indices/full_set/training/"
    data_indices_dir = dataset_dir + "indices/full_set/all/"
    groups_dir = dataset_dir + "groups/"


    def __init__(self, markov_order=0, training_indices_dir=None, data_indices_dir=None, alpha=1):
        if training_indices_dir is not None:
            Classifier.training_indices_dir = training_indices_dir
        if data_indices_dir is not None:
            Classifier.data_indices_dir = data_indices_dir

        self.markov_order = markov_order
        self.alpha = alpha
        self.start_tokens = []
        for i in xrange(self.markov_order):
            self.start_tokens.append("__start%d__" % i)


    def train(self):
        """
        Train the training classes based on the training data
        """
        self.training_classes = {}
        self.initclasses(self.training_classes, Classifier.training_indices_dir)
        self.counttypes(self.training_classes)
        for c in self.training_classes:
            c.calcposterior(self.alpha)


    def initclasses(self, classes, indices_dir):
        """
        Initialize the given classes pointing to the
        indices in indices_dir to find data
        """
        indices = [x for x in os.listdir(indices_dir) if not x.startswith('.')]
        for x in indices:
            class_name = self.index_to_class_name(re.sub('\..*', '', x))
            c = classes[class_name] = Class(class_name)
            file_name = indices_dir + x
            c.files.extend(self.getfiles(file_name))


    def getfiles(self, index_file_name):
        """
        Gets the file names of the data
        """
        files = []
        with open(index_file_name) as f:
            for line in f:
                files.append(Classifier.dataset_dir + line.rstrip('\n'))
        return files


    def index_to_class_name(self, class_string):
        return re.sub('_', '.', class_string)


    def counttypes(self, classes):
        """
        Gets the counts for all token types in all classes
        """
        for name in classes:
            c = classes[name]
            for f in c.files:
                c.incdoccount()
                tokens = self.start_tokens + self.tokenize(f)
                for i in xrange(self.markov_order, len(tokens)):
                    token = tokens[i]
                    for j in xrange(1, self.markov_order + 1):
                        token = tokens[i - j] + " " + token
                    c.addtoken(token)


    def tokenize(self, file_path):
        """
        Takes the contents of a data file and tokenizes the relevant content
        """
        tokens = []
        with open(file_path) as f:
            metadata = True
            for line in f:
                if metadata and line.startswith('Lines: '): # ignore metadata
                    metadata = False
                elif not metadata: # split the tokens into words
                    tokens.extend(''.join(c for c in line if c.isalnum() or c.isspace()).split())
        return tokens


    def classify(self):
        self.data_classes = {}
        self.initclasses(self.data_classes, Classifier.data_indices_dir)
        self.counttypes(self.data_classes)


    stopwords = ["a", "and", "or", "of", "as", "be", "but", "does", "for"]



"""
Classy class that represents a class of random variable... class...
"""
class Class:


    all_types = sets.Set()
    all_doc_count = 0


    def __init__(self, name):
        self.name = name
        self.files = []
        self.types = {}
        self.total_tokens = 0
        self.doc_count = 0
        self.posterior = {}


    def addtoken(self, token):
        if not token in self.types:
            self.types[token] = 0
        self.types[token] += 1
        self.total_tokens += 1
        Class.all_types.add(token)


    def calcposterior(self, alpha=1):
        d = (alpha - 1) * len(Class.all_types)
        for type_name in Class.all_types:
            if not type_name in self.types: # make sure they are all there
                self.types[type_name] = 0

            # alpha is for MAP smoothing
            freq = float(self.types[type_name] + alpha - 1)
            self.posterior[type_name] = freq / (self.total_tokens + d)


    def incdoccount(self):
        self.doc_count += 1
        Class.all_doc_count += 1


    def gettypecount(self, type_name):
        if type_name in self.types:
            return self.types[type_name]
        return 0


    def gettotaltypes(self):
        return len(self.types)


    def gettotaltokens(self):
        return self.total_tokens


    def getposteriorat(self, type_name):
        if type_name not in self.posterior:
            return 1
        return self.posterior




"""
Representation of a document being classified
"""
class Document:


    def __init__(self, actual_class=None):
        self.actual_class = actual_class
        self.likelihoods = {}
        self.success = None


    def calclikelihood(self, class_i, tokens):
        likelihood = float(class_i.doc_count) / Class.all_doc_count
        for token in tokens
            likelihood = likelihood * class_i.getposteriorat(token)

        likelihoods[class_i.name] = likelihood


    def guessclass(self):
        max = 0
        guess = None
        for l in likelihoods:
            if likelihoods[l] > max:
                max = likelihoods[i]
                guess = l

        if self.actual_class is not None:
            if guess == self.actual_class:
                self.success = True
            else
                self.success = False
        return guess


"""
Main function if run as executable
"""
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classifies documents from the 20 Newsgroups dataset")
    parser.add_argument('-m', '--markov-order', type=int, default=0)
    parser.add_argument('-t', '--training-indices-dir')
    parser.add_argument('-d', '--data-indices-dir')
    parser.add_argument('-a', '--alpha', type=int, default=1, help="for symmetrical MAP smoothing of posterior")
    args = parser.parse_args()
    print args

    if 'training_indices_dir' in args:
        training_indices_dir = args.training_indices_dir
    if 'data_indices_dir' in args:
        data_indices_dir = args.data_indices_dir
    c = Classifier(args.markov_order, training_indices_dir, data_indices_dir, args.alpha)

    print c.indices
