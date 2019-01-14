from sklearn.cluster import KMeans
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def carregar_imagens_kmeans(path):
	fileList = glob.glob(path)
	fileList.sort()
	imgList = []

	for infile in fileList:
		img = Image.open(infile)
		img = np.array(img)
		img = [pixel for linha in img for pixel in linha]
		imgList.append(img)

	return imgList

def get_labels(path, k):
	lista_imagens = carregar_imagens_kmeans(path)

	kmeans = KMeans(n_clusters = k, init = 'random')
	kmeans.fit(lista_imagens)

	return kmeans.labels_

def get_melhor_k():
	lista_imagens = carregar_imagens_kmeans('output/*.jpg')

	# kmeans = KMeans(n_clusters = 3, init = 'random')
	# kmeans.fit(lista_imagens)

	wcss = []
	 
	for i in range(1, 11):
	    kmeans = KMeans(n_clusters = i, init = 'random')
	    kmeans.fit(lista_imagens)
	    print(i,kmeans.inertia_)
	    wcss.append(kmeans.inertia_)

	plt.plot(range(1, 11), wcss)
	plt.title('Metodo Elbow')
	plt.xlabel('Numero de Clusters')
	plt.ylabel('WSS') #within cluster sum of squares
	plt.show()

	print(kmeans.labels_)