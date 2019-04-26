import os
import random
from timeit import default_timer as timer
from star_reader import read_stars
from tree_bootstrapped import Tree
from forest import Forest


OUTPUT_FOLDER = "output/forest"


def log(s, open_file):
    print(s)
    open_file.write(str(s) + '\n')


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    if not os.path.exists(OUTPUT_FOLDER + "/testing.txt"):
        output = open(OUTPUT_FOLDER + "/testing.txt", 'w',
                      encoding="utf-8")
    else:
        output = open(OUTPUT_FOLDER + "/testing.txt", 'a', encoding="utf-8")

    dataset, fields = read_stars()

    random.shuffle(dataset)

    cutoff = 0.25
    forest_size = 10

    split = int(len(dataset) * cutoff)
    training, testing = dataset[:split], dataset[split + 1:]

    log("\n----------\n", output)

    log("\n-- TREE TRAINING --\n", output)

    log("Training Tree...", output)
    t_start = timer()

    log("Dataset split: Training with {}% of the set".format(cutoff*100), output)
    log("Training set: {} entries.".format(len(training)), output)
    log("Testing set: {} entries.".format(len(testing)), output)

    tree = Tree(fields, training, [i for i in range(len(training))])

    t_end = timer()
    log("Training complete.\nElapsed time: {:.3f}\n".format(t_end - t_start), output)

    log("\n-- TREE TEST --\n", output)

    total_success = 0

    for entry in testing:
        success, predict = tree.predict(entry)
        # print("Actual: {}\tPredicted: {}.\tSuccess: {}".format(entry.label, predict, success))
        total_success += success

    tested = len(testing)
    s_rate = float(total_success)*100/float(tested)

    log("\nTested {} entries.".format(tested), output)

    log("Accuracy: {:.2f}%\nError: {:.2f}%".format(s_rate, 100-s_rate), output)

    log("\n-- FOREST TRAINING --\n", output)

    log("Training Forest...", output)
    t_start = timer()

    log("Dataset split: Training with {}% of the set".format(cutoff*100), output)
    log("Training set: {} entries.".format(len(training)), output)
    log("Testing set: {} entries.".format(len(testing)), output)

    forest = Forest(fields, training, forest_size)

    t_end = timer()
    log("Training complete.\nElapsed time: {:.3f}\n".format(t_end - t_start), output)

    log("\n-- FOREST TEST --\n", output)

    total_success = 0

    for entry in testing:
        label = entry.label
        majority = forest.predict(entry)
        if majority in label:
            # print("Actual: {}\tPredicted: {}".format(label, predict))
            total_success += 1

    tested = len(testing)
    s_rate = float(total_success)*100/float(tested)

    log("\nTested {} entries.".format(tested), output)

    log("Accuracy: {:.2f}%\nError: {:.2f}%".format(s_rate, 100-s_rate), output)

    error = forest.error_oob()

    log("\nError Out-of-Bag: {:.2f}%".format(error*100), output)

    output.close()
