from preprocessor import Preprocessor
from classifier import Classifier
import sys


def main():
    batch_size = int(sys.argv[1])
    save_model = True if sys.argv[2] == "True" else False

    preprocessor = Preprocessor(batch_size)
    preprocessor.load()
    preprocessor.show_examples()

    training_loader = preprocessor.training_loader
    test_loader = preprocessor.test_loader
    classes = preprocessor.classes
    batch_size = preprocessor.batch_size

    classifier = Classifier()
    classifier.train(training_loader, save_model)
    classifier.evaluate(test_loader, classes, batch_size)


if __name__ == '__main__':
    main()
