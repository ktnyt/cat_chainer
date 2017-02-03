import chainer
import chainer.links as L

class SLP(chainer.Chain):
    def __init__(self):
        super(SLP, self).__init__(
            layer=L.Linear(None, 10),
        )

    def __call__(self, x):
        return self.layer(x)
