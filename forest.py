import random
import operator
from tree_bootstrapped import Tree


class Forest(object):
    def __init__(self, fields, dataset, size, tree_out=False, out=True):
        self.fields = fields
        self.dataset = dataset
        self.size = size

        self.trees = []
        for i in range(size):
            n = len(dataset)
            bootstrap = [random.randrange(n) for j in range(n)]
            tree = Tree(self.fields, self.dataset, bootstrap, (tree_out and out))
            self.trees.append(tree)

            if out:
                print("\nPlanted tree {}".format(i))

    def error_oob(self):
        oob = []
        for tree in self.trees:
            oob.extend(tree.oob)

        oob = set(oob)

        votes = {}
        successes = 0

        for i in oob:
            entry = self.dataset[i]

            for tree in self.trees:
                if i not in tree.indices:
                    predict = tree.classify(entry).predictions
                    for key, value in predict.items():
                        if key not in votes:
                            votes[key] = predict[key]
                        else:
                            votes[key] += predict[key]
                    majority = max(votes.items(), key=operator.itemgetter(1))[0]
                    if majority in entry.label:
                        successes += 1

        return 1-(float(successes)/float(len(oob)))

    def predict(self, entry):
        votes = {}
        for tree in self.trees:
            predict = tree.classify(entry).predictions
            for key, value in predict.items():
                if key not in votes:
                    votes[key] = predict[key]
                else:
                    votes[key] += predict[key]
        majority = max(votes.items(), key=operator.itemgetter(1))[0]
        return majority
