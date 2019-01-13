import math, time
import random
from random import randint

def distancia_euclidiana(p, q):
	distancia = 0
	
	for i in range(len(p)):
		distancia += (p[i] - q[i])**2
	distancia = math.sqrt(distancia)

	return distancia

def distancia_hamming(p, q):
	distancia = 0

	for i in range(len(p)):
		if(p[i] != q[i]):
			distancia += 1

	return distancia

def media_lista(lista):
	tam = len(lista)
	soma = 0
	for i in lista: soma += i

	return float(soma)/float(tam)

def desvio_padrao(lista):
	tam = len(lista)
	media = media_lista(lista)
	soma = 0
	for i in lista: soma += (i - media)**2

	desvio = math.sqrt(float(soma)/float(tam))

	return desvio

def ordenar_lista_maior(lista):
	if(len(lista) >= 2):
		for i in range(len(lista)-1,0,-1):
			if(lista[i].distancia > lista[i-1].distancia):
				lista[i],lista[i-1] = lista[i-1],lista[i]

	return lista

def print_lista(lista):
	print('[')
	for i in lista:
		print('\t'+str(i))
	print(']')

def get_k_proximos(lista, elemento, k, funcao_distancia=distancia_euclidiana):
	lista_proximos = []

	for j,q in enumerate(lista):
		distancia = funcao_distancia(elemento.vetor_carac,q.vetor_carac)

		if(len(lista_proximos) < k):
			lista_proximos.append(q)
			q.distancia = distancia
		elif(distancia < lista_proximos[0].distancia):
			lista_proximos.pop()
			lista_proximos.append(q)
			q.distancia = distancia
			ordenar_lista_maior(lista_proximos)

	return lista_proximos

def get_proximos_limiar(lista, elemento, limiar, funcao_distancia=distancia_hamming):
	lista_proximos = []

	for i,q in enumerate(lista):
		distancia = funcao_distancia(elemento.img_hash,q.img_hash)

		if(distancia < limiar):
			lista_proximos.append(q)

	return lista_proximos
