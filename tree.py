import multiprocessing as mp
from question import Question


def unique_vals(dataset, column):
    return set([entry.data[column] for entry in dataset])


def count_labels(dataset):
    counts = {}
    for entry in dataset:
        for label in entry.label:
            if label not in counts:
                counts[label] = 1
            else:
                counts[label] += 1
    return counts


def partition(dataset, question):
    matching, non_matching = [], []

    for entry in dataset:
        if question.match(entry):
            matching.append(entry)
        else:
            non_matching.append(entry)

    return matching, non_matching


def gini(dataset):
    counts = count_labels(dataset)
    impurity = 1

    for label in counts:
        prob = counts[label] / float(len(dataset))
        impurity -= prob**2

    return impurity


def info_gain(left_set, right_set, uncertainty):
    p = float(len(left_set)) / float(len(left_set) + len(right_set))

    return uncertainty - p * gini(left_set) - (1-p) * gini(right_set)


def splitter(info):
    question, dataset, uncertainty = info
    matching, non_matching = partition(dataset, question)
    if not matching or not non_matching:
        return None
    gain = info_gain(matching, non_matching, uncertainty)
    return (gain, question, (matching, non_matching))


def find_best_split(fields, dataset, uncertainty=None):
    print("Splitting {} entries.".format(len(dataset)))
    best_gain, best_question, best_split = 0, None, None

    uncertainty = uncertainty or gini(dataset)

    columns = len(dataset[0].data)

    for i in range(columns):
        values = unique_vals(dataset, i)

        if len(dataset) > 400:
            # Parallelize best split search
            cpus = mp.cpu_count()
            if i == 0:
                print("-- Using {} CPUs to parallelize the split search.".format(cpus))
            splits = []
            for value in values:
                question = Question(fields, i, value)
                splits.append((question, dataset, uncertainty))

            chunk = max(int(len(splits)/(cpus*4)), 1)
            with mp.Pool(cpus) as p:
                for split in p.imap_unordered(splitter, splits, chunksize=chunk):
                    if split is not None:
                        gain, question, branches = split
                        if gain > best_gain:
                            best_gain, best_question, best_split = gain, question, branches
        else:
            for value in values:
                question = Question(fields, i, value)

                matching, non_matching = partition(dataset, question)

                if not matching or not non_matching:
                    continue

                gain = info_gain(matching, non_matching, uncertainty)

                if gain > best_gain:
                    best_gain, best_question = gain, question
                    best_split = (matching, non_matching)

    return best_gain, best_question, best_split


class Node(object):
    def __init__(self, fields, dataset, level=0):
        self.fields = fields
        self.gini = gini(dataset)
        self.build(dataset, level)

    def build(self, dataset, level):
        best_split = find_best_split(self.fields, dataset, self.gini)
        gain, question, branches = best_split

        if not branches:
            # Means we got 0 gain
            print("Found a leaf at level {}".format(level))
            self.predictions = count_labels(dataset)
            self.is_leaf = True
            return

        left, right = branches

        print("Found a level {} split:".format(level))
        print(question)
        print("Matching: {} entries\tNon-matching: {} entries".format(len(left), len(right)))

        self.left_branch = Node(self.fields, left, level + 1)
        self.right_branch = Node(self.fields, right, level + 1)
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

    def predict(self, entry):
        successes = []
        predict = self.classify(entry).predictions.copy()

        for label in entry.label:
            total = float(sum(predict.values()))
            for key, value in predict.items():
                predict[key] = float(predict[key]) / total
            if label in predict:
                success = float(predict[label]) / total
                successes.append(success)

        return sum(successes), predict

    def print(self, spacing=''):
        if self.is_leaf:
            s = spacing + "Predict: "
            total = float(sum(self.predictions.values()))
            probs = {}
            for label in self.predictions:
                prob = self.predictions[label] * 100 / total
                probs[label] = "{:.2f}%".format(prob)
            return s + str(probs)

        s = spacing + "(Gini: {:.2f}) ".format(self.gini) + str(self.question) + '\n'
        s += spacing + "├─ True:\n"
        s += self.left_branch.print(spacing + "│  ") + '\n'
        s += spacing + "└─ False:\n"
        s += self.right_branch.print(spacing + "   ")

        return s

    def __str__(self):
        return self.print()


class Tree(object):
    def __init__(self, fields, dataset):
        self.fields = fields
        self.dataset = dataset
        self.root = Node(self.fields, self.dataset)

    def classify(self, entry):
        return self.root.classify(entry)

    def predict(self, entry):
        return self.root.predict(entry)

    def __str__(self):
        return str(self.root)
