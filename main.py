from preprocessor import Preprocessor
from classifier import Classifier


def main():
    preprocessor = Preprocessor(batch_size=4)
    preprocessor.load()
    preprocessor.show_examples()

    training_loader = preprocessor.training_loader
    test_loader = preprocessor.test_loader
    classes = preprocessor.classes
    batch_size = preprocessor.batch_size

    classifier = Classifier()
    classifier.train(training_loader)
    classifier.evaluate(test_loader, classes, batch_size)


if __name__ == '__main__':
    main()
