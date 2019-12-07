from regression import Regression

r1)egression = Regression()
regression.load_abalone_dataset()
regression.init_model()
regression.train_and_test(epoch_count=100, mb_size=10, report=