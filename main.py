from regression import Regression

regression = Regression()
regression.load_abalone_dataset()
regression.init_model()
regression.train_and_test(epoch_count=100, mb_size=10, report=1)