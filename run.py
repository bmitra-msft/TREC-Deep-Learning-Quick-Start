from learner import Learner
from utils import Utils, Printer


def main():
    printer = Printer('log.txt')
    utils = Utils(printer)
    utils.setup_and_verify()
    utils.evaluate_baseline()
    learner = Learner(utils.learner_utils)
    learner.train_and_evaluate()
    utils.printer.print('finished!')

if __name__ == '__main__':
    main()
