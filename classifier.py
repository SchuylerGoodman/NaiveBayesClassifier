import os
import os.path
import re
import sets
import math
from time import sleep

class Classifier:

    """
    default class variables
    """
    dataset_dir = "20_newsgroups/"
    training_indices_dir = dataset_dir + "indices/full_allDev/training/"
    data_indices_dir = dataset_dir + "indices/full_allDev/dev/"
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
        print "Training classifier on indices in {0}".format(Classifier.training_indices_dir)
        self.training_classes = {}
        self.initclasses(self.training_classes, Classifier.training_indices_dir)
        self.counttypes(self.training_classes)
        for c in self.training_classes:
            class_i = self.training_classes[c]
            class_i.calcestimates(self.alpha)


    def initclasses(self, classes, indices_dir):
        """
        Initialize the given classes with paths to the data
        files pointed to by indices_dir
        """
        print "Initializing Class objects"
        indices = [x for x in os.listdir(indices_dir) if not x.startswith('.')]
        for x in indices:
            class_name = self.file_to_class_name(x)
            c = classes[class_name] = Class(class_name)
            file_name = indices_dir + x
            c.files.extend([Classifier.dataset_dir + x for x in self.getfiles(file_name)])


    def getfiles(self, index_file_name):
        """
        Gets the file names of the data
        """
        print "Getting data file names for index {0}".format(index_file_name)
        files = []
        with open(index_file_name) as f:
            for line in f:
                files.append(line.rstrip('\n'))
        return files


    def file_to_class_name(self, file_name):
        return self.index_to_class_name(re.sub('\.txt', '', file_name))


    def index_to_class_name(self, class_string):
        return re.sub('_', '.', class_string)


    def counttypes(self, classes):
        """
        Gets the counts for all token types in all classes
        """
        print "Counting word types"
        for name in classes:
            c = classes[name]
            num = len(c.files)
            count = 0
            for f in c.files:
                c.incdoccount()
                tokens = self.start_tokens + self.tokenize(f)
                for i in xrange(self.markov_order, len(tokens)):
                    token = tokens[i]
                    for j in xrange(1, self.markov_order + 1):
                        token = tokens[i - j] + " " + token
                    c.addtoken(token)
                count += 1
                if count % 500 == 0 or count == num:
                    print "{0} of {1} files counted for class {2}".format(count, num, c.name)


    def tokenize(self, file_path):
        """
        Takes the contents of a data file and tokenizes the relevant content
        """
        tokens = []
#        print "Tokenizing file at {0}".format(file_path)
        with open(file_path) as f:
            metadata = True
            for line in f:
                if metadata and line.startswith('Lines: '): # ignore metadata
                    metadata = False
                elif not metadata: # split the tokens into words
                    tokens.extend(''.join(c for c in line if c.isalnum() or c.isspace()).split())
#        print "{0} tokens found".format(len(tokens))
        return tokens


    def classify(self):
        print "Classifying documents in {0}".format(Classifier.data_indices_dir)
        files = []
        self.documents = {}
        indices = [x for x in os.listdir(Classifier.data_indices_dir) if not x.startswith('.')]
        for x in indices:
            file_name = Classifier.data_indices_dir + x
            files.extend(self.getfiles(file_name))

        print "{0} files found for classification".format(len(files))
        for f in files:
            if f in self.documents:
                continue
            raw_tokens = self.start_tokens + self.tokenize(Classifier.dataset_dir + f)
            tokens = []
            for i in xrange(self.markov_order, len(raw_tokens)):
                token = raw_tokens[i]
                for j in xrange(1, self.markov_order + 1):
                    token = raw_tokens[i - j] + " " + token
                tokens.append(token)
            self.documents[f] = Document(f, tokens, self.file_to_class_name(f).split('/')[1])

        num = len(self.documents)
        count = 0
        for name in self.documents:
#            print "Classifying document {0}".format(name)
            doc = self.documents[name]
            for c in self.training_classes:
                class_i = self.training_classes[c]
                doc.calcllikelihood(class_i)
            doc.guessclass()
            count += 1
            if (count % 1000 == 0):
                print "Classified {0} out of {1}".format(count, num)

        return self.documents


    def classifyandprint(self):
        docs = self.classify()
        success = 0
        none = 0
        failure = 0
        confusion = {}
        for name in docs:
            doc = docs[name]
            for c in self.training_classes:
                if c not in confusion:
                    confusion[c] = [[], {}]
                class_i = self.training_classes[c]
                if doc.actual_class == class_i.name:
                    if doc.guess_class == class_i.name:
                        confusion[c][0].append(doc)
                    else:
                        if doc.guess_class not in confusion[c][1]:
                            confusion[c][1][doc.guess_class] = []
                        confusion[c][1][doc.guess_class].append(doc)

            if doc.success == True:
                success += 1
            elif doc.success is None:
                none += 1
            else:
                failure += 1
            print "file: {3}  actual: {0}  guess: {1}  success: {2}"\
                    .format(doc.actual_class, doc.guess_class, doc.success, doc.name)
        percent = float(success) / (len(docs) - none)
        print "success: {0}  failure: {1}  none: {2}  percent: {3}"\
                .format(success, failure, none, percent)
        for c in confusion:
            confused = confusion[c]
            print "Documents properly labeled {0} : {1}".format(c, len(confused[0]))
            for c2 in confused[1]:
                reallyconfused = confused[1][c2]
                print "Documents labeled {0} improperly labeled {1} : {2}".format(c, c2, len(reallyconfused))


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
        self.total_tokens = 0
        self.class_types = sets.Set()
        self.doc_count = 0
        self.distributions = {}
        self.junk_param = None


    def addtoken(self, token):
        stokens = token.rsplit(' ', 1) # split if we have an n-gram
        if len(stokens) > 1:
            context = stokens[0]
            token = stokens[1]
        else:
            context = "__all__"
            token = stokens[0] # not really necessary, but I like the look more

        if context not in self.distributions: # set up different contexts for the dist. over vocab
            self.distributions[context] = {}
        dist = self.distributions[context]
        if token not in dist:
            dist[token] = [token, 0, 0]
        dist[token][1] = dist[token][1] + 1
        self.total_tokens += 1
        self.class_types.add(token)
        Class.all_types.add(token)


    def calcestimates(self, alpha=1):
        Sd = (alpha - 1) * len(Class.all_types)
        denom = self.total_tokens + Sd
        num = len(Class.all_types)
        count = 0
        for type_name in Class.all_types:
            for context in self.distributions:
                dist = self.distributions[context]
                if not type_name in dist: # make sure they are all there
                    continue
#                    dist[type_name] = [type_name, 0, 0]

                # alpha is for MAP smoothing
                freq = float(dist[type_name][1] + alpha - 1)
                if freq == 0.0:
                    dist[type_name][2] = float("-inf")
                else:
                    dist[type_name][2] = math.log(freq / denom)
            count += 1
            if count % 1000 == 0 or count == num:
                print "{0} out of {1} type name estimates calculated for class {2}".format(count, num, self.name)
            if count == num:
                print "sleeping to cool down"
                sleep(180)
                print "starting up"

        if self.junk_param is None:
            junk_freq = float(alpha - 1)
            if junk_freq == 0.0:
                self.junk_param = float("-inf")
            else:
                self.junk_param = math.log(freq / denom)


    def incdoccount(self):
        self.doc_count += 1
        Class.all_doc_count += 1


    def getestimateat(self, type_name, context="__all__"):
        if context in self.distributions:
            dist = self.distributions[context]
            if type_name in dist:
                return dist[type_name][2]
        return self.junk_param



"""
Representation of a document being classified
"""
class Document:


    def __init__(self, name, tokens, actual_class=None):
        self.name = name
        self.tokens = tokens
        self.actual_class = actual_class
        self.guess_class = None
        self.llikelihoods = {}
        self.success = None


    def calcllikelihood(self, class_i):
        llikelihood = math.log(float(class_i.doc_count) / Class.all_doc_count)
        num = len(self.tokens)
        count = 0
        for token in self.tokens:
            stoken = token.rsplit(' ', 1)
            context = "__all__"
            if len(stoken) > 1:
                token = stoken[1]
                context = stoken[0]
            llikelihood = llikelihood + class_i.getestimateat(token, context)
            if num % 10000 == 0 or num == count:
                print "{0} out of {1} log likelihoods calculated for document {2}. Current tokens: {3}".format(count, num, self.name, stoken)

        self.llikelihoods[class_i.name] = llikelihood


    def guessclass(self):
        max = -float("inf")
        guess = None
        for l in self.llikelihoods:
            if self.llikelihoods[l] > max:
                max = self.llikelihoods[l]
                guess = l

        if self.actual_class is not None:
            if guess == self.actual_class:
                self.success = True
            else:
                self.success = False
        self.guess_class = guess
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
    parser.add_argument('-a', '--alpha', type=int, default=1, help="for symmetrical uniform smoothing of MAP estimate distributions")
    args = parser.parse_args()
    print args

    if 'training_indices_dir' in args:
        training_indices_dir = args.training_indices_dir
    if 'data_indices_dir' in args:
        data_indices_dir = args.data_indices_dir
    c = Classifier(args.markov_order, training_indices_dir, data_indices_dir, args.alpha)
    c.train()
    c.classifyandprint()
