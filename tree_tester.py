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
        output = open("output/tree_testing.txt", 'w', encoding="utf-8")
    else:
        output = open("output/tree_testing.txt", 'a', encoding="utf-8")

    dataset, fields = read_stars()

    log("\n----------\n", output)

    log("Training Tree...", output)
    t_start = timer()

    split = int(len(dataset) * 0.65)
    training, testing = dataset[:split], dataset[split + 1:]
    log("Training set: {} entries.".format(len(training)), output)
    log("Testing set: {} entries.".format(len(testing)), output)

    tree = Tree(fields, training)

    t_end = timer()
    timestamp = "Training complete.\nElapsed time: {:.3f}\n"
    log(timestamp.format(t_end - t_start), output)

    log(tree, output)

    log("\n-- TEST --\n", output)

    failures = 0

    for entry in testing:
        label = entry.label
        predict = tree.predict(entry)
        if predict not in label:
            print("Actual: {}\tPredicted: {}".format(label, predict))
            failures += 1

    tested = len(testing)
    success = tested - failures
    s_rate = float(success)*100/float(tested)

    log("\nSuccessfully predicted {} out of {} entries."
        .format(success, tested), output)

    log("Accuracy: {:.2f}%\nError: {:.2f}%".format(s_rate, 100-s_rate), output)

    output.close()
