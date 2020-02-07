import numpy as np
#import cupy as np


class MLP:
    def init_model_hidden1(self):
        self.pm_hidden = self.alloc_param_pair([self.input_cnt, self.hidden_cnt])
        self.pm_output = self.alloc_param_pair([self.hidden_cnt, self.output_cnt])

    def alloc_param_pair(self, shape):
        self.weight = np.random.normal(self.RND_MEAN, self.RND_STD, shape)
        self.bias = np.zeros(shape[-1])
        return {'w': self.weight, 'b': self.bias }

    def forward_neuralnet_hidden1(self, x):
        self.hidden = self.relu(np.matmul(x, self.pm_hidden['w']) + self.pm_hidden['b'])
        self.output = np.matmul(self.hidden, self.pm_output['w']) + self.pm_output['b']

        return self.output, [x, self.hidden]

    def relu(self, x):
        return np.maximum(x, 0)

    def backprop_neuralnet_hidden1(self, G_output, aux):
        x, hidden = aux

        g_output_w_out = self.hidden.transpose()
        G_w_out = np.matmul(g_output_w_out, G_output)
        G_b_out = np.sum(G_output, axis=0)

        g_output_hidden = self.pm_output['w'].transpose()
        G_hidden = np.matmul(G_output, g_output_hidden)

        self.pm_output['w'] -= self.LEARNING_RATE * G_w_out
        self.pm_output['b'] -= self.LEARNING_RATE * G_b_out

        G_hidden = G_hidden * self.relu_derv(hidden)

        g_hidden_w_hid = x.transpose()
        G_w_hid = np.matmul(g_hidden_w_hid, G_hidden)
        G_b_hid = np.sum(G_hidden, axis=0)

        self.pm_hidden['w'] -= self.LEARNING_RATE * G_w_hid
        self.pm_hidden['b'] -= self.LEARNING_RATE * G_b_hid

    def relu_derv(self, y):
        return np.sign(y)

    def init_model_hiddens(self):
        self.pm_hiddens = []
        prev_cnt = self.input_cnt

        for hidden_cnt in self.hidden_config:
            self.pm_hiddens.append(self.alloc_param_pair([prev_cnt, hidden_cnt]))
            prev_cnt = hidden_cnt

        self.pm_output = self.alloc_param_pair([prev_cnt, self.output_cnt])

    def forward_neuralnet_hiddens(self, x):
        hidden = x
        hiddens = [x]

        for pm_hidden in self.pm_hiddens:
            hidden = self.relu(np.matmul(hidden, pm_hidden['w'] + pm_hidden['b']))
            hiddens.append(hidden)

        output = np.matmul(hidden, self.pm_output['w']) + self.pm_output['b']

        return output, hiddens

    def backprop_neuralnet_hiddens(self, G_output, aux):
        hiddens = aux

        g_output_w_out = hiddens[-1].transpose()
        G_w_out = np.matmul(g_output_w_out, G_output)
        G_b_out = np.sum(G_output, axis=0)

        g_output_hidden = self.pm_output['w'].transpose()
        G_hidden = np.matmul(G_output, g_output_hidden)

        self.pm_output['w'] -= self.LEARNING_RATE * G_w_out
        self.pm_output['b'] -= self.LEARNING_RATE * G_b_out

        for n in reversed(range(len(self.pm_hiddens))):
            G_hidden = G_hidden * self.relu_derv(hiddens[n+1])

            g_hidden_w_hid = hiddens[n].transpose()
            G_w_hid = np.matmul(g_hidden_w_hid, G_hidden)
            G_b_hid = np.sum(G_hidden, axis=0)

            g_hidden_hidden = self.pm_hiddens[n]['w'].transpose()
            G_hidden = np.matmul(G_hidden, g_hidden_hidden)

            self.pm_hiddens[n]['w'] -= self.LEARNING_RATE * G_w_hid
            self.pm_hiddens[n]['b'] -= self.LEARNING_RATE * G_b_hid

    def init_model(self):
        if self.hidden_config is not None:
            print('은닉 계층 {}개를 갖는 다층 퍼셉트론이 작동되었습니다.'. \
                  format(len(self.hidden_config)))
            self.init_model_hiddens()
        else:
            print('은닉 계층 하나를 갖는 다층 퍼셉트론이 작동되었습니다.')
            self.init_model_hidden1()

    def forward_neuralnet(self, x):
        if self.hidden_config is not None:
            return self.forward_neuralnet_hiddens(x)
        else:
            return self.forward_neuralnet_hidden1(x)

    def backprop_neuralnet(self, G_output, hiddens):
        if self.hidden_config is not None:
            self.backprop_neuralnet_hiddens(G_output, hiddens)
        else:
            self.backprop_neuralnet_hidden1(G_output, hiddens)

    def set_hidden(self, info):
        if isinstance(info, int):
            self.hidden_cnt = info
            self.hidden_config = None
        else:
            self.hidden_config = info
