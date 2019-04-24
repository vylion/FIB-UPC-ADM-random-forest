import multiprocessing as mp
from question import Question


def unique_vals(dataset, indices, column):
    return set([dataset[i].data[column] for i in indices])


def count_labels(dataset, indices):
    counts = {}
    for i in indices:
        for label in dataset[i].label:
            if label not in counts:
                counts[label] = 1
            else:
                counts[label] += 1
    return counts


def partition(dataset, indices, question):
    matching, non_matching = [], []

    for i in indices:
        if question.match(dataset[i]):
            matching.append(i)
        else:
            non_matching.append(i)

    return matching, non_matching


def gini(dataset, indices):
    counts = count_labels(dataset, indices)
    impurity = 1

    for label in counts:
        prob = counts[label] / float(len(dataset))
        impurity -= prob**2

    return impurity


def info_gain(dataset, lid, rid, uncertainty):
    p = float(len(lid)) / float(len(lid) + len(rid))

    return uncertainty - p * gini(dataset, lid) - (1-p) * gini(dataset, rid)


def splitter(info):
    question, dataset, indices, uncertainty = info
    matching, non_matching = partition(dataset, indices, question)
    if not matching or not non_matching:
        return None
    gain = info_gain(dataset, matching, non_matching, uncertainty)
    return (gain, question, (matching, non_matching))


def find_best_split(fields, dataset, indices, uncertainty=None):
    print("Splitting {} entries.".format(len(dataset)))
    best_gain, best_question, best_split = 0, None, None

    uncertainty = uncertainty or gini(dataset)

    columns = len(fields)

    for i in range(columns):
        values = unique_vals(dataset, indices, i)

        if len(indices) > 400:
            # Parallelize best split search
            cpus = mp.cpu_count()
            if i == 0:
                print("-- Using {} CPUs to parallelize the split search."
                      .format(cpus))
            splits = []
            for value in values:
                question = Question(fields, i, value)
                splits.append((question, dataset, indices, uncertainty))

            chunk = max(int(len(splits)/(cpus*4)), 1)
            with mp.Pool(cpus) as p:
                for split in p.imap_unordered(splitter, splits,
                                              chunksize=chunk):
                    if split is not None:
                        gain, question, branches = split
                        if gain > best_gain:
                            best_gain, best_question, best_split = \
                                gain, question, branches
        else:
            for value in values:
                question = Question(fields, i, value)

                matching, non_matching = partition(dataset, indices, question)

                if not matching or not non_matching:
                    continue

                gain = info_gain(dataset, matching, non_matching, uncertainty)

                if gain > best_gain:
                    best_gain, best_question = gain, question
                    best_split = (matching, non_matching)

    return best_gain, best_question, best_split


class Node(object):
    def __init__(self, fields, dataset, bootstrap, level=0):
        self.fields = fields
        self.dataset = dataset
        self.bootstrap = bootstrap
        self.gini = gini(dataset, self.bootstrap)
        self.build(level)

    def build(self, level):
        best_split = find_best_split(self.fields, self.dataset,
                                     self.bootstrap, self.gini)
        gain, question, branches = best_split

        if not branches:
            # Means we got 0 gain
            print("Found a leaf at level {}".format(level))
            self.predictions = count_labels(self.dataset, self.bootstrap)
            self.is_leaf = True
            return

        left, right = branches

        print("Found a level {} split:".format(level))
        print(question)
        print("Matching: {} entries\tNon-matching: {} entries".format(len(left), len(right))) # noqa

        self.left_branch = Node(self.fields, self.dataset, left, level + 1)
        self.right_branch = Node(self.fields, self.dataset, right, level + 1)
        self.question = question
        self.is_leaf = False
        return

    def classify(self, entry):
        if self.is_leaf:
            return self

        if self.question.match(entry):
            return self.left_branch.classify(entry)
        else:
            return self.right_branch.classify(entry)

    def print(self, spacing=''):
        if self.is_leaf:
            s = spacing + "Predict: "
            total = float(sum(self.predictions.values()))
            probs = {}
            for label in self.predictions:
                prob = self.predictions[label] * 100 / total
                probs[label] = "{:.2f}%".format(prob)
            return s + str(probs)

        s = spacing + ("(Gini: {:.2f}) {}\n"
                       .format(self.gini, str(self.question)))
        s += spacing + "├─ True:\n"
        s += self.left_branch.print(spacing + "│  ") + '\n'
        s += spacing + "└─ False:\n"
        s += self.right_branch.print(spacing + "│  ")

        return s

    def __str__(self):
        return self.print()


class Tree(object):
    def __init__(self, fields, dataset, bootstrap):
        self.fields = fields
        self.dataset = dataset
        self.bootstrap = bootstrap
        self.root = Node(self.fields, self.dataset, self.bootstrap)

    def classify(self, entry):
        return self.root.classify(entry)

    def __str__(self):
        return str(self.root)
