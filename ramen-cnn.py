import numpy as np
import chainer
from chainer import cuda, Function, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import dataset
import pickle
import os.path



class MyModel(Chain):
	def __init__(self):
		super(MyModel, self).__init__(
			cn1 = L.Convolution2D(3,20,5), # size : 160 - 5 + 1 = 156 
			cn2 = L.Convolution2D(20, 50, 5), # size : 52 - 5  + 1 = 48
			l1 = L.Linear(800, 100),#45 * 45 * 50
			# cn1 = L.Convolution2D(1,20,5), # size : 160 - 5 + 1 = 156 , 20 * 3 | (60, 156, 156)
			# cn2 = L.Convolution2D(20,50,5), # size : 39 - 5 + 1 = 35 | (60, 35, 35)
			# l1 = L.Linear(50, 25),
			l2 = L.Linear(100, 7),
			)
	#def __call__(self, x, t):
	#	return F.softmax_cross_entropy(self.fwd(x), t)

	def __call__(self, x):
		#print(x.data)
		#print(x.data.shape)
		#print(x.get_example())
		h1 = F.max_pooling_2d(F.relu(self.cn1(x)), 4) # pool1: 55 - 4 + 1 = 52
		h2 = F.max_pooling_2d(F.relu(self.cn2(h1)), 4) # pool2: 48 - 4 + 1 = 45
		h3 = F.dropout(F.relu(self.l1(h2)))
		return self.l2(h3)


if __name__=='__main__':
	model = L.Classifier(MyModel())
	# model = MyModel()
	if os.path.isfile('./dataset.pickle'):
		print("dataset.pickle is exist. loading...")
		with open('./dataset.pickle', mode='rb') as f:
			train, test = pickle.load(f)
		print("Loaded")
	else:
		datasets = dataset.Dataset("sampleimages", 80, 80)
		train, test = datasets.get_dataset()
		with open('./dataset.pickle', mode='wb') as f:
			print("saving train and test...")
			pickle.dump((train,test), f)



	# train, test = datasets.get_mnist(ndim=3)
	optimizer = chainer.optimizers.Adam()
	optimizer.setup(model)
	train_iter = chainer.iterators.SerialIterator(train, 32)
	test_iter = chainer.iterators.SerialIterator(test, 32, repeat=False, shuffle=False)

	updater = training.StandardUpdater(train_iter, optimizer, device=-1)
	trainer = training.Trainer(updater, (50, 'epoch'), out="logs")

	trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
	trainer.extend(extensions.LogReport())
	trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
	trainer.extend(extensions.ProgressBar())
	trainer.run()
	print("Learn END")






