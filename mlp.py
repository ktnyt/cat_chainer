import chainer
import chainer.functions as F
import chainer.links as L

class MLP(chainer.Chain):
    def __init__(self):
        super(MLP, self).__init__(
            l1=L.Linear(None, 1000),
            l2=L.Linear(None, 1000),
            l3=L.Linear(None, 10),
        )

    def __call__(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        return self.l3(h2)
