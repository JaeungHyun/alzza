import numpy as np
#import cupy as np
import pandas as pd
from regression import Regression
import csv
from mlp import MLP


class PerceptronSoftmax(Regression, MLP):
    def load_steel_dataset(self):
        ## pandas로 불러오는 것 적용하기
        with open('data/chap03/faults.csv') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader, None)
            rows = []
            for row in csvreader:
                rows.append(row)

        self.input_cnt, self.output_cnt = 27, 7
        self.data = np.asarray(rows, dtype='float32')

    def forward_postproc(self, output, y):
        entropy = self.softmax_cross_entropy_with_logits(y, output)
        loss = np.mean(entropy)

        return loss, [y, output, entropy]

    def backprop_postproc(self, G_loss, aux):
        y, output, entropy = aux

        # For debugging
        print(entropy.shape)
        g_loss_entropy = 1.0 / np.prod(entropy.shape)
        g_entropy_output = self.softmax_cross_entropy_with_logits_derv(y, output)

        G_entropy = g_loss_entropy * G_loss
        G_output = g_entropy_output * G_entropy

        return G_output

    def eval_accuracy(self, output, y):
        estimate = np.argmax(output, axis=1)
        answer = np.argmax(y, axis=1)
        correct = np.equal(estimate, answer)

        return np.mean(correct)

    def softmax(self, x):
        max_elem = np.max(x, axis=1)
        diff = (x.transpose() - max_elem).transpose()
        exp = np.exp(diff)
        sum_exp = np.sum(exp, axis=1)
        probs = (exp.transpose() / sum_exp).transpose()
        return probs

    def softmax_derv(self, x, y):
        mb_size, nom_size = x.shape
        derv = np.ndarray([mb_size, nom_size, nom_size])
        for n in range(mb_size):
            for i in range(nom_size):
                for j in range(nom_size):
                    derv[n, i, j] = -y[n, i] * y[n, j]
                derv[n, i, i] += y[n, i]
        return derv

    def softmax_cross_entropy_with_logits(self, labels, logits):
        probs = self.softmax(logits)
        return -np.sum(labels * np.log(probs + 1.0e-10), axis=1)

    def softmax_cross_entropy_with_logits_derv(self, labels, logits):
        return self.softmax(logits) - labels

    def steel_exec(self, epoch_count=10, mb_size=10, report=1):
        self.load_steel_dataset()
        self.init_model()
        self.train_and_test(epoch_count, mb_size, report)


if __name__ == "__main__":
    perceptronSoftmax = PerceptronSoftmax([12, 6, 4])
    perceptronSoftmax.steel_exec(epoch_count=100, report=10)