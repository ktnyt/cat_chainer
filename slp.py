import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import Variable, FunctionSet, optimizers
import chainer.functions  as F

class SLP(FunctionSet):
    def __init__(self, n_input, n_output):
        super(SLP, self).__init__(
            transform=F.Linear(n_input, n_output)
        )

    def forward(self, x_data, y_data):
        x = Variable(x_data)
        t = Variable(y_data)
        y = F.sigmoid(self.transform(x))
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

mnist = fetch_mldata('MNIST original')
mnist.data = mnist.data.astype(np.float32)
mnist.data /= 255
mnist.target = mnist.target.astype(np.int32)

batchsize = 100

N = 60000

x_train, x_test = np.split(mnist.data, [N])
y_train, y_test = np.split(mnist.target, [N])
N_test = y_test.size

model = SLP(28 ** 2, 10)
optimizer = optimizers.Adam()
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
