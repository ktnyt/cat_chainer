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

optimizer0 = optimizers.Adam()

class DAE(FunctionSet):
    def __init__(self, n_input, n_hidden):
        super(DAE, self).__init__(
            encoder=F.Linear(n_input, n_hidden),
            decoder=F.Linear(n_hidden, n_input)
        )

    def forward(self, x_data):
        x = Variable(x_data)
        t = Variable(x_data)
        h = F.sigmoid(self.encoder(x))
        y = F.sigmoid(self.decoder(h))
        return F.mean_squared_error(y, t)

    def encode(self, x_data):
        x = Variable(x_data)
        h = F.sigmoid(self.encoder(x))
        return h.data

    def decode(self, h_data):
        h = Variable(h_data)
        y = F.sigmoid(self.decoder(h))
        return y.data

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

model0 = DAE(28 ** 2, 1000)
model0.to_gpu()
optimizer0.setup(model0.collect_parameters())

for epoch in xrange(10):
    print "Epoch {}".format(epoch + 1)

    perm = np.random.permutation(N)

    sum_loss = 0

    for i in xrange(0, N, batchsize):
        x_batch = x_train[perm[i:i+batchsize]]
        optimizer0.zero_grads()
        loss = model0.forward(x_batch)
        loss.backward()
        optimizer0.update()
        sum_loss += loss.data * batchsize

    mean_loss = sum_loss / N

    print "Train:\tLoss={}".format(mean_loss)

    sum_loss = 0

    for i in xrange(0, N_test, batchsize):
        x_batch = x_test[i:i+batchsize]
        loss = model0.forward(x_batch)
        sum_loss += loss.data * batchsize

    mean_loss = sum_loss / N_test

    print "Test:\tLoss={}".format(mean_loss)

optimizer1 = optimizers.Adam()
model1 = SLP(1000, 10)
model1.to_gpu()
optimizer1.setup(model1.collect_parameters())

model0.to_cpu()
x_train = model0.encode(x_train)
x_test = model0.encode(x_test)

for epoch in xrange(10):
    print "Epoch {}".format(epoch + 1)

    perm = np.random.permutation(N)

    sum_loss = 0
    sum_acc = 0

    for i in xrange(0, N, batchsize):
        x_batch = x_train[perm[i:i+batchsize]]
        y_batch = y_train[perm[i:i+batchsize]]
        optimizer1.zero_grads()
        loss, acc = model1.forward(x_batch, y_batch)
        loss.backward()
        optimizer1.update()
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
        loss, acc = model1.forward(x_batch, y_batch)
        sum_loss += loss.data * batchsize
        sum_acc += acc.data * batchsize

    mean_loss = sum_loss / N_test
    mean_acc = sum_acc / N_test

    print "Test:\tLoss={}\tAcc={}".format(mean_loss, mean_acc)
