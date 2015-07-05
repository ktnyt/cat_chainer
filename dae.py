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

model = DAE(28 ** 2, 1000)
optimizer.setup(model.collect_parameters())

for epoch in xrange(10):
    print "Epoch {}".format(epoch + 1)

    perm = np.random.permutation(N)

    sum_loss = 0

    for i in xrange(0, N, batchsize):
        x_batch = x_train[perm[i:i+batchsize]]
        optimizer.zero_grads()
        loss = model.forward(x_batch)
        loss.backward()
        optimizer.update()
        sum_loss += loss.data * batchsize

    mean_loss = sum_loss / N

    print "Train:\tLoss={}".format(mean_loss)

    sum_loss = 0

    for i in xrange(0, N_test, batchsize):
        x_batch = x_test[i:i+batchsize]
        loss = model.forward(x_batch)
        sum_loss += loss.data * batchsize

    mean_loss = sum_loss / N_test

    print "Test:\tLoss={}".format(mean_loss)
