import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import glob
import numpy
import matplotlib.pyplot as plt

from util import *
from kmeans import *

qnt_classes = 3
lista_rotulos = []
limiar = 3
k = 6

# camadas de saida
out1=3
out2=8
out3=11

# tamanho dos filtros
f1=4
f2=4
f3=4

# stride dos filtros
s1 = 1
s2 = 1
s3 = 1

# tamanho das matrizes MaxPolling
m1=2
m2=2
m3=2

# tamanho das camadas fully connected
tam_fc1 = 200
tam_fc2 = 20
tam_fc3 = qnt_classes

# tamanho da imagem na CNN
tam_img=100

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

		self.max1 = nn.MaxPool2d(m1)
		self.max2 = nn.MaxPool2d(m2)
		self.max3 = nn.MaxPool2d(m3)

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
		x = self.max1(x)
		x = F.relu(self.conv2(x))
		x = self.max2(x)
		x = F.relu(self.conv3(x))
		x = self.max3(x)

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


class Imagem():
	def __init__(self, imagem, classe, img_hash, vetor_carac):
		self.imagem = imagem
		self.classe = classe
		self.img_hash = img_hash
		self.vetor_carac = vetor_carac
		self.distancia = 0


def salvar_rotulos(lista):
	arq = open('classes.data','w')

	for r in lista:
		arq.write(str(r)+'\n')

	arq.close()

def carregar_rotulos():
	arq = open('classes.data','r')

	linhas = arq.readlines()

	arq.close()

	rotulos = []

	for l in linhas:
		rotulos.append(int(l.replace('\n','')))

	return rotulos

def carregar_imagens_cnn(path):
	fileList = glob.glob(path)
	fileList.sort()
	tensorList = []

	for infile in fileList:
		im = Image.open(infile)
		img = numpy.array(im)
		img_tensor = preprocess(im)
		img_tensor.unsqueeze_(0)
		tensorList.append(img_tensor)

	return tensorList

def carregar_bases():
	# carregando imagens
	baixa_qualidade = carregar_imagens_cnn("input/*.jpg")
	alta_qualidade = carregar_imagens_cnn("output/*.jpg")

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

def criar_cnn(particoes, net_dados=None):
	if(net_dados == None):
		net = Net()
		print(net)

		optimizer = optim.SGD(net.parameters(), lr=0.01)

		ultimaValidacao = 1
		validacaoUnderFitting = 0.5

		listaLoss = []
		listaLossValid = []

		melhor_cnn_dados = {'epoch':1,
							'model_state_dict': None,
							'optimizer_state_dict': None,
							'loss': 0.5}
		melhor_cnn = Net()

		epoch = 0
	else:
		net = Net()
		net.load_state_dict(net_dados['model_state_dict'])
		print(net)

		optimizer = optim.SGD(net.parameters(), lr=0.01)
		optimizer.load_state_dict(net_dados['optimizer_state_dict'])

		ultimaValidacao = net_dados['loss']
		validacaoUnderFitting = net_dados['loss']+0.01

		# listaLoss = net_dados['listaLoss']
		# listaLossValid = net_dados['listaLossValid']

		listaLoss = []
		listaLossValid = []

		melhor_cnn_dados = net_dados

		melhor_cnn = net

		epoch = net_dados['epoch']

	# arq = open('continuar.data','r')
	# continuar = int(arq.readline())
	# arq.close()

	# while(continuar):
	for i in range(30):
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

		print('epoca=%d\t\tloss=%.7f\tvalidacao=%.7f\tDiferenca=%.7f' % ((epoch+1), valueLoss, valueValidLoss, (ultimaValidacao-valueLoss)))

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

		if(ultimaValidacao < melhor_cnn_dados['loss']):
			melhor_cnn_dados['epoch'] = epoch
			melhor_cnn_dados['model_state_dict'] = net.state_dict()
			melhor_cnn_dados['optimizer_state_dict'] = optimizer.state_dict()
			melhor_cnn_dados['loss'] = ultimaValidacao
			melhor_cnn.load_state_dict(net.state_dict())

		# arq = open('continuar.data','r')
		# continuar = int(arq.readline())
		# arq.close()

		epoch += 1

	salvar = input('Salvar rede neural? (s/n) ')
	if(salvar == 's'):
		nome_arq = input('Digite o nome do arquivo: ')
		melhor_cnn_dados['listaLoss'] = listaLoss
		melhor_cnn_dados['listaLossValid'] = listaLossValid
		torch.save(melhor_cnn_dados, nome_arq+'.pth')

	return melhor_cnn,melhor_cnn_dados,[listaLoss,listaLossValid]

def testar_cnn(net, aprendizado, net_dados, particoes):
	epoca = net_dados['epoch']

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
	global lista_rotulos

	###### PARA GERAR NOVOS ROTULOS
	lista_rotulos = get_labels('output/*.jpg', 3)
	salvar_rotulos(lista_rotulos)

	###### PARA USAR ROTULOS GRAVADOS
	# lista_rotulos = carregar_rotulos()

	particoes = carregar_bases()

	###### CARREGAR REDE
	# net_dados = torch.load('rede_240_1.pth')

	###### INICIAR NOVA REDE
	net_dados = None

	net,net_dados,aprendizado = criar_cnn(particoes, net_dados)

	testar_cnn(net, aprendizado, net_dados, particoes)

main()