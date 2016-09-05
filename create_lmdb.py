import glob
import argparse
import os
import lmdb
import random
import numpy as np
import math

from PIL import Image

from caffe.io import array_to_datum

IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227
SCORE = 6

def createDB(name, contenido, funcion):
	db = lmdb.Environment(name, map_size=int(1e12))
	tx = db.begin(write=True)

	for label, imagen in enumerate(contenido):
		im = Image.open(imagen)
		punt = im.fp
		im = im.resize((IMAGE_WIDTH,IMAGE_HEIGHT))
		if 'cat' in os.path.split(imagen)[1]:
			y = 0
		else:
			y = 1
		x = np.array(im.getdata()).reshape(im.size[1],im.size[0],3)
		datum = array_to_datum(np.transpose(x,(2,0,1)),y)

		if funcion(label,SCORE):
			print label
			tx.put('{:08}'.format(label),datum.SerializeToString())

		if not punt.closed:
			punt.close()

		if (label+1) % 2500 == 0:
			tx.commit()
			tx = db.begin(write=True)
			print '------- commit -------'

	tx.commit()
	db.close()

def argumentos():
	parser = argparse.ArgumentParser(description = 'Creacion lmdb training and validation')

	parser.add_argument('-r', '--ruta', type=str, help='Ruta de imagenes', required=True)
	parser.add_argument('-d', '--destino', type=str, help='Ruta destiono', required=False, default=None)

	args = parser.parse_args()

	if args.destino:
		return args.ruta, args.destino
	else:
		return args.ruta, args.ruta

ruta, destino = argumentos()
ruta = os.path.abspath(ruta)
destino = os.path.abspath(destino)

print ruta, destino

if os.path.exists(ruta) and os.path.exists(destino):
	contenido = glob.glob(ruta + '/' + '*.jpg')

	random.shuffle(contenido)

	'''
	imagenes = []
	for i in contenido:
		imagenes.append(os.path.split(i)[1])
	#print imagenes

	print imagenes
	'''

	print 'Creando lmdb train'
	createDB(destino + '/train_lmdb',contenido,lambda x, y: x % y != 0) # lmdb train
	print 'Done.'

	print 'Creando lmdb validation'
	createDB(destino + '/validation_lmdb',contenido,lambda x, y: x % y == 0) # lmdb validation
	print 'Done.'
