import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import glob, os
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import random

from util import *

qnt_classes = 8
lista_rotulos = [random.randint(0,qnt_classes-1) for i in range(909)]
limiar = 2
k = 6

# camadas de saida
out1=1
out2=1
out3=1

# tamanho dos filtros
f1=1
f2=1
f3=1

# stride dos filtros
s1 = 2
s2 = 2
s3 = 1

# tamanho das matrizes MaxPolling
m1=2
m2=2
m3=2

# tamanho das camadas fully connected
tam_fc1 = 90
tam_fc2 = 25
tam_fc3 = qnt_classes

# tamanho da imagem na CNN
tam_img=256

# transformacoes que serao aplicadas as imagens
normalize = transforms.Normalize(
	mean=[0.485, 0.456, 0.406],
	std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
	transforms.Resize(tam_img),
	transforms.CenterCrop(tam_img-10),
	transforms.ToTensor(),
	normalize
])

# criterio da loss function
criterion = nn.MSELoss()


class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()

		# conv layers
		self.conv1 = nn.Conv2d(1, out1, f1, s1)
		self.conv2 = nn.Conv2d(out1, out2, f2, s2)
		self.conv3 = nn.Conv2d(out2, out3, f3, s3)

		# fully connected layers
		saida_cnn = self.get_tam_saida_cnn()
		self.fc1 = nn.Linear(saida_cnn, tam_fc1)
		self.fc2 = nn.Linear(tam_fc1, tam_fc2)
		self.fc3 = nn.Linear(tam_fc2, tam_fc3)

		self.camada_vetor_carac = []
		self.camada_hash = []

	def forward(self, x):

		## CNN

		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x,m1)
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x,m2)
		x = F.relu(self.conv3(x))
		x = F.max_pool2d(x,m3)

		## FULLY CONNECTED

		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))

		self.camada_vetor_carac = x

		x = torch.sigmoid(self.fc2(x))

		self.camada_hash = x 

		x = F.relu(self.fc3(x))

		return x

	def mostrar_camadas(self):
		for camada in self.saida_das_camadas:
			np_array=camada
			np_array=np_array[0,:,:,:]
			np_array=np_array.detach().numpy()
			for i in range(1):
				for j in range(camada[0].size()[0]):
					plt.subplot2grid((1, camada[0].size()[0]), (i, j)).imshow(np_array[i+j])

			plt.show()

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

	def calcular_saida_camada_conv(self, tam_input, tam_filtro, stride):
		tam = int((tam_input-tam_filtro+1)/float(stride))
		return tam

	def get_tam_saida_cnn(self):
		out_camada_1 = int(self.calcular_saida_camada_conv(tam_img-10, f1, s1)/float(m1))
		out_camada_2 = int(self.calcular_saida_camada_conv(out_camada_1, f2, s2)/float(m2))
		out_camada_3 = int(self.calcular_saida_camada_conv(out_camada_2, f3, s3)/float(m3))
		return out_camada_3 * out_camada_3 * out3

	def get_hash(self):
		lista_hash = []
		for instance in self.camada_hash:
			lista = instance.data.tolist()
			lista_hash.append([])
			for output in lista:
				if(output >= 0.5):
					lista_hash[-1].append(1)
				else:
					lista_hash[-1].append(0)
		return lista_hash

	def get_vetor_carac(self):
		lista_vetor_carac = []
		for instance in self.camada_vetor_carac:
			lista_vetor_carac.append(instance.data.tolist())
		return lista_vetor_carac


def carregar_imagens(path):
	fileList = glob.glob(path)
	tensorList = []

	for infile in fileList:
		im = Image.open(infile)
		# img = np.array(im)
		# print(img[0][0])
		# exit()
		img_tensor = preprocess(im)
		img_tensor.unsqueeze_(0)
		tensorList.append(img_tensor)

	return tensorList

def carregar_bases():
	# carregando imagens
	baixa_qualidade = carregar_imagens("input/*.jpg")
	alta_qualidade = carregar_imagens("output/*.jpg")

	particoes = {}
	particoes['treino'] = {'imagens':[] , 'expected':[] , 'hash':[], 'vetor_carac':[]}
	particoes['teste'] = {'imagens':[] , 'expected':[] , 'hash':[], 'vetor_carac':[]}
	particoes['validacao'] = {'imagens':[] , 'expected':[] , 'hash':[], 'vetor_carac':[]}

	# separar as imagens por rotulo
	imagens_por_classe = [[] for i in range(qnt_classes)]
	for i in range(len(baixa_qualidade)):
		rotulo = lista_rotulos[i]
		imagens_por_classe[rotulo].append([baixa_qualidade[i] , alta_qualidade[i]])

	# criar os conjuntos de treino, teste e validacao
	for i,classe in enumerate(imagens_por_classe):
		count = 0
		tam = len(classe)

		expected = [0.0 for i in range(qnt_classes)]
		expected[i] = 1.0

		trainSize = int(tam*0.8)
		for img in classe[count:trainSize]:
			particoes['treino']['imagens'].append(img[0])
			particoes['treino']['expected'].append(torch.Tensor([expected]))
			particoes['treino']['imagens'].append(img[1])
			particoes['treino']['expected'].append(torch.Tensor([expected]))
			count += 1

		validSize = int(tam*0.1)
		for img in classe[count:count+validSize]:
			particoes['validacao']['imagens'].append(img[0])
			particoes['validacao']['expected'].append(torch.Tensor([expected]))
			particoes['validacao']['imagens'].append(img[1])
			particoes['validacao']['expected'].append(torch.Tensor([expected]))
			count += 1

		for img in classe[count:]:
			particoes['teste']['imagens'].append(img[0])
			particoes['teste']['expected'].append(torch.Tensor([expected]))
			particoes['teste']['imagens'].append(img[1])
			particoes['teste']['expected'].append(torch.Tensor([expected]))
			count += 1

	# converter as listas em tensores
	for part in particoes:
		for lista in ('imagens','expected'):
			particoes[part][lista] = torch.cat(particoes[part][lista], 0)

	return particoes

def criar_cnn(particoes):
	net = Net()
	print(net)
	print('m1 =',m1,'; m2 =',m2,'m3 =',m3)

	optimizer = optim.SGD(net.parameters(), lr=0.01)

	ultimaValidacao = 1
	validacaoUnderFitting = 0.5

	listaLoss = []
	listaLossValid = []

	melhor_cnn = Net()
	melhor_epoca = 1
	melhor_loss = 0.5

	epoch = 0
	while(True):
		optimizer.zero_grad()   # zero the gradient buffers

		output = net(particoes['treino']['imagens'])
		loss = criterion(output, particoes['treino']['expected'])
		valueLoss = loss.data.tolist()
		listaLoss.append(valueLoss)

		outputValidacao = net(particoes['validacao']['imagens'])
		validLoss = criterion(outputValidacao, particoes['validacao']['expected'])
		valueValidLoss = validLoss.data.tolist()
		listaLossValid.append(valueValidLoss)

		if(epoch == 0):
			ultimaValidacao = valueValidLoss

		if(valueValidLoss > ultimaValidacao):
			print('over fitting')
			# break

		ultimaValidacao = valueValidLoss

		loss.backward()
		optimizer.step()	# Does the update

		if(epoch % 10 == 10-1):
			criterioUnderFitting = 0.007
			if(ultimaValidacao < 0.13):
				criterioUnderFitting = 0.000001
			if(validacaoUnderFitting - ultimaValidacao < criterioUnderFitting):
				print('under fitting')
				# break
			validacaoUnderFitting = ultimaValidacao

		if(ultimaValidacao < melhor_loss):
			melhor_cnn.load_state_dict(net.state_dict())
			# melhor_cnn = copy.deepcopy(net)
			melhor_loss = ultimaValidacao
			melhor_epoca = epoch

		print((epoch+1), ', loss=', valueLoss, ', validacao=', valueValidLoss)

		arq = open('continuar.data','r')
		continuar = int(arq.readline())
		arq.close()

		if(not continuar):
			break

		epoch += 1

	# return net,ultimaValidacao,[listaLoss,listaLossValid]
	return melhor_cnn,melhor_loss,melhor_epoca,[listaLoss,listaLossValid]


class Imagem():
	def __init__(self, imagem, classe, img_hash, vetor_carac):
		self.imagem = imagem
		self.classe = classe
		self.img_hash = img_hash
		self.vetor_carac = vetor_carac
		self.distancia = 0


def testar_cnn(net, aprendizado, epoca, particoes):
	# calcular os hash e vetor de caracteristicas de todas as imagens
	base_imagens = torch.cat([particoes['treino']['imagens'], particoes['validacao']['imagens']], 0)

	base_classes = torch.cat([particoes['treino']['expected'], particoes['validacao']['expected']], 0)

	net(base_imagens)

	base_classes = base_classes.data.tolist()

	base_hash = net.get_hash()

	base_vetor_carac = net.get_vetor_carac()

	base_treino = []
	for i in range(len(base_imagens)):
		base_treino.append(Imagem(base_imagens[i], base_classes[i], base_hash[i], base_vetor_carac[i]))

	net(particoes['teste']['imagens'])

	teste_hash = net.get_hash()

	teste_vetor_carac = net.get_vetor_carac()

	base_teste = []
	for i in range(len(particoes['teste']['imagens'])):
		base_teste.append(Imagem(particoes['teste']['imagens'][i], particoes['teste']['expected'][i], teste_hash[i], teste_vetor_carac[i]))

	# for img in base_teste:
	# 	proximos = get_proximos_limiar(base_treino, img, limiar)
	# print(len(proximos))

	proximos = get_proximos_limiar(base_treino, base_teste[0], limiar)

	semelhantes = get_k_proximos(base_treino, base_teste[0], k)

	plt.plot(aprendizado[0][:epoca])
	plt.plot(aprendizado[1][:epoca])
	plt.show()

	img = base_teste[0].imagem
	img = img[0,:]
	img = img.detach().numpy()
	plt.subplot2grid((1, 1), (0, 0)).imshow(img, cmap="gray")
	plt.show()

	for i in range(2):
		for j in range(3):
			ind_img = i*3+j

			img = semelhantes[ind_img].imagem
			img = img[0,:]
			img = img.detach().numpy()
			plt.subplot2grid((2, 3), (i, j)).imshow(img, cmap="gray")
	plt.show()

def main():
	particoes = carregar_bases()

	net,loss,epoca,aprendizado = criar_cnn(particoes)

	testar_cnn(net, aprendizado, epoca, particoes)

main()