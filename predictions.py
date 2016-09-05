import argparse
import os
import numpy as np
import glob
from PIL import Image

import caffe
from caffe.io import caffe_pb2, blobproto_to_array, Transformer

MODEL_FILE = 'caffenet_train_iter_10000.caffemodel'
DEPLOY_FILE = 'deploy.prototxt'
MEAN_FILE = 'mean.binaryproto'
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

def argumentos():
	parser = argparse.ArgumentParser(description = 'Predicciones de imagenes')

	parser.add_argument('-r', '--ruta', type=str, help='Ruta de imagenes TEST', required=True)

	args = parser.parse_args()

	return args.ruta

ruta = os.path.abspath(argumentos())

print ruta

if os.path.exists(ruta):
	imagenes = glob.glob(ruta + '/' + '[0-1][0-9].jpg')

	#print imagenes

	fp = open(MEAN_FILE,'rb')
	proto_data = fp.read()
	fp.close()

	mean_blob = caffe_pb2.BlobProto()
	mean_blob.ParseFromString(proto_data)

	mean_array = blobproto_to_array(mean_blob)[0]

	net = caffe.Net(DEPLOY_FILE,MODEL_FILE,caffe.TEST)

	transformer = Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_mean('data', mean_array)
	transformer.set_transpose('data', (2,0,1))

	'''
	net = caffe.Classifier(DEPLOY_FILE,MODEL_FILE,
       mean=mean_array,
       channel_swap=(2,1,0),
       raw_scale=255,
       image_dims=(IMAGE_WIDTH,IMAGE_HEIGHT))
    '''

	labels = []
	predic = []
	for n,i in enumerate(imagenes):
		img = Image.open(i)
		fp = img.fp
		img = img.resize((IMAGE_WIDTH,IMAGE_HEIGHT))
		x = np.array(img.getdata()).reshape(img.size[1],img.size[0],3)

		net.blobs['data'].data[...] = transformer.preprocess('data',x)
		out = net.forward()
		pred_probas = out['prob']

		labels += [os.path.split(i)[1]]
		predic += [pred_probas.argmax()]

		print labels[n], predic[n]

		if not fp.closed:
			fp.close()
