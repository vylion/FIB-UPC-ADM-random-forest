import os
from timeit import default_timer as timer
from star_reader import read_stars
from tree import Tree


def log(s, open_file):
    print(s)
    open_file.write(str(s) + '\n')


if __name__ == '__main__':
    if not os.path.exists("output"):
        os.mkdir("output")

    if not os.path.exists("output/tree_testing.txt"):
        output = open("output/tree_testing.txt", 'w')
    else:
        output = open("output/tree_testing.txt", 'a')

    dataset, fields = read_stars()

    log("\n----------\n", output)

    log("Training Tree...", output)
    t_start = timer()

    split = int(len(dataset) * 0.65)
    split = 500
    training, testing = dataset[:split], dataset[split + 1:]
    log("Training set: {} entries.".format(len(training)), output)
    log("Testing set: {} entries.".format(len(testing)), output)

    tree = Tree(fields, training)

    t_end = timer()
    timestamp = "Training complete.\nElapsed time: {:.3f}\n"
    log(timestamp.format(t_end - t_start), output)

    log(tree, output)

    log("\n-- TEST --\n", output)

    for entry in testing:
        label = entry.label
        predict = tree.classify(entry)
        log("Actual: {}\tPredicted: {}".format(label, predict), output)

    output.close()
