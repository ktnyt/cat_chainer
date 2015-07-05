import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import Variable, FunctionSet, optimizers
import chainer.functions  as F

mnist = fetch_mldata('MNIST original')
mnist.data = mnist.data.astype(np.float32)
mnist.data /= 255
mnist.target = mnist.target.astype(np.int32)

batchsize = 100

N = 60000

x_train, x_test = np.split(mnist.data, [N])
y_train, y_test = np.split(mnist.target, [N])
N_test = y_test.size

optimizer = optimizers.Adam()

class MLP(FunctionSet):
    def __init__(self, n_input, n_hidden, n_output):
        super(MLP, self).__init__(
            l1=F.Linear(n_input, n_hidden),
            l2=F.Linear(n_hidden, n_output)
        )

    def forward(self, x_data, y_data):
        x = Variable(x_data)
        t = Variable(y_data)
        h = F.sigmoid(self.l1(x))
        y = F.sigmoid(self.l2(h))
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

model = MLP(28 ** 2, 500, 10)
optimizer.setup(model.collect_parameters())

for epoch in xrange(10):
    print "Epoch {}".format(epoch + 1)

    perm = np.random.permutation(N)

    sum_loss = 0
    sum_acc = 0

    for i in xrange(0, N, batchsize):
        x_batch = x_train[perm[i:i+batchsize]]
        y_batch = y_train[perm[i:i+batchsize]]
        optimizer.zero_grads()
        loss, acc = model.forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()
        sum_loss += loss.data * batchsize
        sum_acc += acc.data * batchsize

    mean_loss = sum_loss / N
    mean_acc = sum_acc / N

    print "Train:\tLoss={}\tAcc={}".format(mean_loss, mean_acc)

    sum_loss = 0
    sum_acc = 0

    for i in xrange(0, N_test, batchsize):
        x_batch = x_test[i:i+batchsize]
        y_batch = y_test[i:i+batchsize]
        loss, acc = model.forward(x_batch, y_batch)
        sum_loss += loss.data * batchsize
        sum_acc += acc.data * batchsize

    mean_loss = sum_loss / N_test
    mean_acc = sum_acc / N_test

    print "Test:\tLoss={}\tAcc={}".format(mean_loss, mean_acc)
