import numpy as np
from collections import Counter
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# classe nó
class Node:
    def __init__(self, atributo=None, value=None, resp=None, true_branch=None, false_branch=None):
        self.atributo = atributo         # índice do atributo usado para fazer a divisão nesse nó
        self.value = value               # valor de corte para o atributo
        self.resp = resp                 # resposta/classificação do nó folha
        self.true_branch = true_branch   # subárvore para <= valor
        self.false_branch = false_branch # subárvore para > valor


class ID3:
    def __init__(self):
        self.arvore = None
        self.nomes_atributos = None

    # função para calcular entropia
    def calcular_entropia(self, y):
        # quantas vezes cada classe aparece
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        # fórmula da entropia: -[somatório( p * log2(p))]
        entropia = 0.0
        for p in probs:
            if p > 0:
                entropia -= p * np.log2(p)
        return entropia

    # função para dividir os dados
    def dividir_dados(self, X, y, atributo, corte):
        true_indices = [] # vetor de lista com os indices verdadeiros
        false_indices = [] # vetor de lista com os indices falsos
        for i in range(len(X)):
            # fazer o corte para cada atributo
            if X[i][atributo] <= corte:
                true_indices.append(i) #adicionar o indice do atributo true na lista
            else:
                false_indices.append(i) #adicionar o indice do atributo true

        true_X = X[true_indices]
        true_y = y[true_indices]
        false_X = X[false_indices]
        false_y = y[false_indices]

        return true_X, true_y, false_X, false_y # retornar tupla com todos os valores

    # função para o ganho de informação
    def ganho_informacao(self, X, y, atributo, corte):
        _, true_y, _, false_y = self.dividir_dados(X, y, atributo, corte) # obter só as parcelas verdadeiras e falsas da divisão
        p_true = len(true_y) / len(y) # parcela verdadeira do total
        p_false = len(false_y) / len(y) # parcela falsa do total
        ganho = self.calcular_entropia(y) - (p_true * self.calcular_entropia(true_y) + p_false * self.calcular_entropia(false_y)) # equação de ganho
        return ganho

    # construção recursiva da árvore
    def construir_arvore(self, X, y):
        # Caso base: todas as classes iguais
        if len(set(y)) == 1:
            return Node(resp=y[0])

        # variáveis "globais" que vão ser atualizadas de acordo com a construção da árvore
        melhor_ganho = 0
        melhor_atributo = None
        melhor_valor = None
        melhor_divisao = None

        # obter o número de atributos
        n_atributos = X.shape[1]

        # percorrer todos os atributos
        for atributo in range(n_atributos):
            # para cada atributo, obter todos as opções de valor dele
            valores_unicos = set(X[:, atributo])
            for valor in valores_unicos:
                # calcular o ganho para cada possível valor do atributo atual (se tiver ganhos iguais, então vai ser o primeiro que aparecer na matriz X)
                ganho = self.ganho_informacao(X, y, atributo, valor)
                if ganho > melhor_ganho:
                    # se encontrar ganho melhor até agora então atualizar os valores globais
                    melhor_ganho = ganho # ganho desse atributo
                    melhor_atributo = atributo # guarda o indice do atributo para ser usado na classificação
                    melhor_valor = valor # valor de corte
                    melhor_divisao = self.dividir_dados(X, y, atributo, valor) # guardar os valores divididos

        # se foi encontrado um ganho continuar dividindo
        if melhor_ganho > 0:
            true_X, true_y, false_X, false_y = melhor_divisao
            true_branch = self.construir_arvore(true_X, true_y) #chamada recursiva para a classificação true
            false_branch = self.construir_arvore(false_X, false_y) #chamada recursiva para a classificação false
            new_no = Node(atributo=melhor_atributo, value=melhor_valor,true_branch=true_branch, false_branch=false_branch) #criar novo nó com essas informações
            return new_no # retornar o nó criado

        # condição de parada da recursão
        # se não houver ganho retorna nó folha
        classe_majoritaria = Counter(y).most_common(1)[0][0] # vai pegar a classe majoritária (se tiver mais "Sim" do que "Não", então a resposta será "Sim")
        new_no = Node(resp=classe_majoritaria) #retornar nó folha com a classe majoritária
        return new_no

    # classificação/predição
    def classificar(self, arvore, amostra):
        # percorrer a árvore com a amostra, até encontrar uma folha
        if arvore.resp is not None: #indica que é folha
            return arvore.resp
        if amostra[arvore.atributo] <= arvore.value:
            return self.classificar(arvore.true_branch, amostra)
        else:
            return self.classificar(arvore.false_branch, amostra)

    # impressão recursiva da árvore
    def imprimir_arvore(self, no, nomes_atributos, espacamento=""):
        if no.resp is not None:
            print(espacamento + "Resposta ->", no.resp)
            return
        nome = nomes_atributos[no.atributo] if no.atributo is not None else "?"
        print(espacamento + f"[{nome} <= {no.value}]")
        print(espacamento + "--> True:")
        self.imprimir_arvore(no.true_branch, nomes_atributos, espacamento + "   ")
        print(espacamento + "--> False:")
        self.imprimir_arvore(no.false_branch, nomes_atributos, espacamento + "   ")

    # função pra mostrar as regras da árvore
    def mostrar_regras(self, no, nomes_atributos, regra_atual=""):
        if no.resp is not None:
            # nó folha → imprime a regra completa com a classe
            print(regra_atual + f" ENTÃO Classe: {no.resp}")
            return
        
        nome = nomes_atributos[no.atributo] if no.atributo is not None else "?"
        
        # caminho True (<= valor)
        nova_regra_true = regra_atual + f"SE {nome} <= {no.value} "
        self.mostrar_regras(no.true_branch, nomes_atributos, nova_regra_true)
        
        # caminho False (> valor)
        nova_regra_false = regra_atual + f"SE {nome} > {no.value} "
        self.mostrar_regras(no.false_branch, nomes_atributos, nova_regra_false)

    # método para treinar (fit)
    def fit(self, X, y, nomes_atributos=None):
        self.nomes_atributos = nomes_atributos
        self.arvore = self.construir_arvore(np.array(X), np.array(y))

    # método para prever (predict)
    def predict(self, X):
        X = np.array(X)
        return np.array([self.classificar(self.arvore, x) for x in X])

    # método para avaliar com métricas
    def avaliar(self, X, y):
        y_pred = self.predict(X)
        acuracia = accuracy_score(y, y_pred)
        precisao = precision_score(y, y_pred, average='macro')
        recall = recall_score(y, y_pred, average='macro')
        f1 = f1_score(y, y_pred, average='macro')

        print(f"Acurácia: {acuracia:.4f}")
        print(f"Precisão: {precisao:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        # matriz de Confusão
        cm = confusion_matrix(y, y_pred, labels=np.unique(y))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Matriz de Confusão - ID3")
        plt.show()

    # salvar modelo
    def salvar(self, caminho="id3.pkl"):
        with open(caminho, "wb") as f:
            pickle.dump(self.arvore, f)

    # carregar modelo
    def carregar(self, caminho="id3.pkl"):
        with open(caminho, "rb") as f:
            self.arvore = pickle.load(f)