import numpy as np
import pandas as pd


class Regression:
    def __init__(self):
        np.random.seed(1234)
        self.RND_MEAN = 0
        self.RND_STD = 0.0030
        self.LEARNING_RATE = 0.001

    def load_abalone_dataset(self):
        df = pd.read_csv('data/chap01/abalone.csv', header=None, skiprows=1)
            
        self.input_cnt, self.output_cnt = 10, 1
        self.data = np.zeros([len(df), self.input_cnt+self.output_cnt])
        
        # 원래있던 sex칼럼을 원핫 인코딩을 적용하여 3 칼럼으로 만들고 나머지 칼럼을 복사해온다.
        for index, row in df.iterrows():
            if row[0] == 'I':
                self.data[index, 0] = 1
            if row[0] == 'M':
                self.data[index, 1] = 1
            if row[0] == 'F':
                self.data[index, 2] = 1
        
            self.data[:, 3:] = df.loc[:, 1:]

    def init_model(self):
        self.weight = np.random.normal(self.RND_MEAN, self.RND_STD, 
                                        [self.input_cnt, self.output_cnt])
        self.bias = np.zeros([self.output_cnt])

    def train_and_test(self, epoch_count, mb_size, report):
        step_count = self.arrange_data(mb_size)
        test_x, test_y = self.get_test_data()

        for epoch in range(epoch_count):
            losses, accs = [], []

            for n in range(epoch_count):
                train_x, train_y = self.get_train_data(mb_size, n)
                loss, acc = self.run_train(train_x, train_y)
                losses.append(loss)
                accs.append(acc)

            if report > 0 and (epoch + 1) % report == 0:
                acc = self.run_test(test_x, test_y)
                print('Epoch {}: loss={:5.3f}, accuracy={:5.3f}/{:5.3f}'.\
                      format(epoch + 1, np.mean(losses), np.mean(accs), acc))

        final_acc = self.run_test(test_x, test_y)
        print('\nFinal Test: final accuracy = {:5.3f}'.format(final_acc))

    def arrange_data(self, mb_size):
        self.shuffle_map = np.arange(self.data.shape[0])
        np.random.shuffle(self.shuffle_map)
        step_count = int(self.data.shape[0] * 0.8) // mb_size
        self.test_begin_idx = step_count * mb_size
        return step_count

    def get_test_data(self):
        test_data = self.data[self.shuffle_map[self.test_begin_idx:]]
        return test_data[:, :-self.output_cnt], test_data[:, -self.output_cnt:]

    def get_train_data(self, mb_size, nth):
        if nth == 0:
            np.random.shuffle(self.shuffle_map[:self.test_begin_idx])
        train_data = self.data[self.shuffle_map[mb_size * nth:mb_size * (nth + 1)]]
        return train_data[:, :-self.output_cnt], train_data[:, -self.output_cnt:]

    def run_train(self, x, y):
        output, aux_nn = self.forward_neuralnet(x)
        loss, aux_pp = self.forward_postproc(output, y)
        accuracy = self.eval_accuracy(output, y)

        G_loss = 1.0
        G_output = self.backprop_postproc(G_loss, aux_pp)
        self.backprop_neuralnet(G_output, aux_nn)

        return loss, accuracy

    def run_test(self, x, y):
        output, _ = self.forward_neuralnet(x)
        accuracy = self.eval_accuracy(output, y)
        return accuracy

    def forward_neuralnet(self, x):
        output = np.matmul(x, self.weight) + self.bias
        return output, x

    def backprop_neuralnet(self, G_output, x):
        g_output_w = x.transpose()

        G_w = np.matmul(g_output_w, G_output)
        G_b = np.sum(G_output, axis=0)

        self.weight -= self.LEARNING_RATE * G_w
        self.bias -= self.LEARNING_RATE * G_b

    def forward_postproc(self, output, y):
        diff = output - y
        square = np.square(diff)
        loss = np.mean(square)
        return loss, diff

    def backprop_postproc(self, G_loss, diff):
        shape = diff.shape

        g_loss_square = np.ones(shape) / np.prod(shape)
        g_square_diff = 2 * diff
        g_diff_output = 1

        G_square = g_loss_square * G_loss
        G_diff = g_square_diff * G_square
        G_output = g_diff_output * G_diff

        return G_output

    def backprop_postproc_oneline(self, diff):
        return 2 * diff / np.prod(diff.shape)

    def eval_accuracy(self, output, y):
        mdiff = np.mean(np.abs((output - y) / y))
        return 1 - mdiff
