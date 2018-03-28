#encoding:UTF-8
import sys
import argparse
import glob
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Variable, optimizers, Chain, datasets, training
from chainer.training import extensions

parser = argparse.ArgumentParser(description='Chainer 3D Autoencoder')
parser.add_argument('--epoch', '-e', type=int, default=20,help='エポック数')
parser.add_argument('--dir', '-d', help='読み込むディレクトリ', default='data01')
parser.add_argument('--dropout', '-r', default=0.25, type=float, help='Dropoutの割合')
args = parser.parse_args()
epoch = args.epoch
dirname = args.dir

# 学習フォルダのディレクトリを指定する
files = glob.glob(dirname + "/*.npy")
train = []
for f in files:
    try:
        ct_data = np.load(f)
        ct_data = ct_data.astype(np.float32) # chainer use float32
    except:
        print("cannot open : " + f)
    train.append(ct_data)


# データセットの作成
train = datasets.TupleDataset(train,train)
train_iter = chainer.iterators.SerialIterator(train, 100)

# ネットワークの定義
class Autoencoder(Chain):
    def __init__(self, drop=False):
        super(Autoencoder, self).__init__()
        with self.init_scope():
                self.conv1 = L.ConvolutionND(3, 1, 3, 3)
                self.dcnv1 = L.DeconvolutionND(3, 3, 1, 3)
        self.drop = drop

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_nd(h,(2,2,2))
        h = F.unpooling_nd(h,(2,2,2))
        h = F.relu(self.dcnv1(h))
        return h

# モデル作成
model = L.Classifier(Autoencoder(), lossfun=F.mean_squared_error)
model.compute_accuracy = False
model.to_gpu()

# GPUを使う設定
optimizer = optimizers.Adam()
optimizer.setup(model)

# オプティマイザの設定
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

# 学習部分
updater = training.StandardUpdater(train_iter, optimizer, device=0)
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='result')
trainer.extend(extensions.LogReport())
#trainer.extend(extensions.Evaluator(test_iter, model, device=0))
trainer.extend(extensions.ProgressBar())
trainer.extend(extensions.PrintReport(['epoch', 'main/loss']))

# Run
trainer.run()

model.to_cpu()
y = model.predictor(train[1][0][None,...])
g = chainer.computational_graph.build_computational_graph(y)
with open('graph.dot', 'w') as f:
        f.write(g.dump())
