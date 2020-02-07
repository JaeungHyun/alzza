import numpy as np
#import cupy as np
import pandas as pd
from regression import Regression
from mlp import MLP


class Perceptron(Regression, MLP):
    def __init__(self, layer):
        super().__init__(layer)

    def load_pulsar_dataset(self, adjust_ratio):
        df = pd.read_csv('data/chap02/pulsar_stars.csv')
        pulsars = np.asarray(df[df['target_class'] == 1])
        stars = np.asarray(df[df['target_class'] == 0])

        self.input_cnt, self.output_cnt = 8, 1

        star_cnt, pulsar_cnt = len(stars), len(pulsars)
        if adjust_ratio:
            self.data = np.zeros([2 * star_cnt, 9])
            self.data[0:star_cnt, :] = np.asarray(stars, dtype='float32')
            for n in range(star_cnt):
                self.data[star_cnt + n] = np.asarray(pulsars[n % pulsar_cnt], dtype='float32')
        else:
            self.data = np.zeros([star_cnt + pulsar_cnt, 9])
            self.data[0:star_cnt, :] = np.asarray(stars, dtype='float32')
            self.data[star_cnt:, :] = np.asarray(pulsars, dtype='float32')

    def forward_postproc(self, output, y):
        entropy = self.sigmoid_cross_entropy_with_logits(y, output)
        loss = np.mean(entropy)
        return loss, [y, output, entropy]

    def backprop_postproc(self, G_loss, aux):
        y, output, entropy = aux

        g_loss_entropy = 1.0 / np.prod(entropy.shape)
        g_entropy_output = self.sigmoid_cross_entropy_with_logits_derv(y, output)

        G_entropy = g_loss_entropy * G_loss
        G_output = g_entropy_output * G_entropy

        return G_output

    def eval_accuracy(self, output, y):
        estimate = np.greater(output, 0)
        answer = np.greater(y, 0.5)
        correct = np.equal(estimate, answer)

        return np.mean(correct)

    def relu(self, x):
        return np.maximum(x, 0)

    def sigmoid(self, x):
        return np.exp(-self.relu(-x)) / (1.0 + np.exp(-np.abs(x)))

    def sigmoid_derv(self, x, y):
        return y * (1 - y)

    def sigmoid_cross_entropy_with_logits(self, z, x):
        return self.relu(x) - x * z + np.log(1 + np.exp(-np.abs(x)))

    def sigmoid_cross_entropy_with_logits_derv(self, z, x):
        return -z + self.sigmoid(x)

    def eval_accuracy(self, output, y):
        est_yes = np.greater(output, 0)
        ans_yes = np.greater(y, 0.5)
        est_no = np.logical_not(est_yes)
        ans_no = np.logical_not(ans_yes)

        tp = np.sum(np.logical_and(est_yes, ans_yes))
        fp = np.sum(np.logical_and(est_yes, ans_no))
        fn = np.sum(np.logical_and(est_no, ans_yes))
        tn = np.sum(np.logical_and(est_no, ans_no))

        accuracy = self.safe_div(tp + tn, tp + tn + fp + fn)
        precision = self.safe_div(tp, tp + fp)
        recall = self.safe_div(tp, tp + fn)
        f1 = 2 * self.safe_div(recall * precision, recall + precision)

        return [accuracy, precision, recall, f1]

    def safe_div(self, p, q):
        p, q = float(p), float(q)
        if np.abs(q) < 1.0e-20:
            return np.sign(p)

        return p / q

    def train_and_test(self, epoch_count, mb_size, report):
        step_count = self.arrange_data(mb_size)
        test_x, test_y = self.get_test_data()

        for epoch in range(epoch_count):
            losses = []

            for n in range(step_count):
                train_x, train_y = self.get_train_data(mb_size, n)
                loss, _ = self.run_train(train_x, train_y)
                losses.append(loss)

            if report > 0 and (epoch + 1) % report == 0:
                acc = self.run_test(test_x, test_y)
                acc_str = ','.join(['%5.3f'] * 4) % tuple(acc)
                print('Epoch {}: loss={:5.3f}, result={}'.format(epoch + 1, np.mean(losses), acc_str))

        acc = self.run_test(test_x, test_y)
        acc_str = ','.join(['%5.3f'] * 4) % tuple(acc)
        print('\nFinal test: final result = {}'.format(acc_str))

    def pulsar_exec(self, epoch_count=10, mb_size=10, report=1, adjust_ratio=False):
        self.load_pulsar_dataset(adjust_ratio)
        self.init_model()
        self.train_and_test(epoch_count, mb_size, report)


if __name__ == "__main__":
    perceptron = Perceptron([6])
    perceptron.pulsar_exec(epoch_count=1000, adjust_ratio=1, report=10)



