import chainer
import chainer.functions as F
import chainer.links as L

class CNN(chainer.Chain):
    def __init__(self):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, 16, 3),
            conv2=L.Convolution2D(16, 32, 3),
            conv3=L.Convolution2D(32, 64, 3),
            l1=L.Linear(None, 512),
            l2=L.Linear(None, 10),
        )

    def __call__(self, x):
        x.data = x.data.reshape((len(x.data), 1, 28, 28))
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, 2)
        h = F.dropout(F.relu(self.l1(h)))
        return self.l2(h)
