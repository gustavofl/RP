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

qnt_classes = 2
lista_rotulos = [random.randint(0,qnt_classes-1) for i in range(909)]

# camadas de saida
out1=3	
out2=3
out3=3

# tamanho dos filtros
f1=3
f2=3
f3=3

# stride dos filtros
s1 = 2
s2 = 1
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

		self.saida_das_camadas = [None for i in range(9)]

	def forward(self, x):

		## CNN

		x = F.relu(self.conv1(x))

		self.saida_das_camadas[0] = x

		x = F.max_pool2d(x,m1)

		self.saida_das_camadas[1] = x

		x = F.relu(self.conv2(x))

		self.saida_das_camadas[2] = x

		x = F.max_pool2d(x,m2)

		self.saida_das_camadas[3] = x

		x = F.relu(self.conv3(x))

		self.saida_das_camadas[4] = x

		x = F.max_pool2d(x,m3)

		self.saida_das_camadas[5] = x

		## FULLY CONNECTED

		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))

		self.saida_das_camadas[6] = x

		x = torch.sigmoid(self.fc2(x))

		self.saida_das_camadas[7] = x 

		x = F.relu(self.fc3(x))

		self.saida_das_camadas[8] = x 

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


def carregar_imagens(path):
	fileList = glob.glob(path)
	tensorList = []

	for infile in fileList:
		im = Image.open(infile)
		img_tensor = preprocess(im)
		img_tensor.unsqueeze_(0)
		tensorList.append(img_tensor)

	return tensorList

def carregar_bases():
	# carregando imagens
	baixa_qualidade = carregar_imagens("input/*.jpg")
	alta_qualidade = carregar_imagens("output/*.jpg")

	particoes = {}
	particoes['treino'] = {'imagens':[] , 'expected':[]}
	particoes['teste'] = {'imagens':[] , 'expected':[]}
	particoes['validacao'] = {'imagens':[] , 'expected':[]}

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
		for lista in particoes[part]:
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

def testar_cnn(net, aprendizado, epoca, particoes):
	output = net(particoes['teste']['imagens'])
	loss = criterion(output, particoes['teste']['expected'])
	print("\n>>> TESTE:",loss.data,"\n")

	plt.plot(aprendizado[0][:epoca])
	plt.plot(aprendizado[1][:epoca])
	plt.show()

def main():
	particoes = carregar_bases()

	net,loss,epoca,aprendizado = criar_cnn(particoes)

	testar_cnn(net, aprendizado, epoca, particoes)

main()