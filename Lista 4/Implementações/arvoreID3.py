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


def calcular_entropia(y):
    # quantas vezes cada classe aparece
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    # fórmula da entropia: -[somatório( p * log2(p))]
    entropia = 0.0
    for p in probs:
        if p > 0:
            entropia -= p * np.log2(p)
    return entropia


def dividir_dados(X, y, atributo, corte):
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
def ganho_informacao(X, y, atributo, corte):
    _, true_y, _, false_y = dividir_dados(X, y, atributo, corte) # obter só as parcelas verdadeiras e falsas da divisão
    p_true = len(true_y) / len(y) # parcela verdadeira do total
    p_false = len(false_y) / len(y) # parcela falsa do total
    ganho = calcular_entropia(y) - (p_true * calcular_entropia(true_y) + p_false * calcular_entropia(false_y)) # equação de ganho
    return ganho


# construção recursiva da árvore
def construir_arvore(X, y):
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
            ganho = ganho_informacao(X, y, atributo, valor)
            if ganho > melhor_ganho:
                # se encontrar ganho melhor até agora então atualizar os valores globais
                melhor_ganho = ganho # ganho desse atributo
                melhor_atributo = atributo # guarda o indice do atributo para ser usado na classificação
                melhor_valor = valor # valor de corte
                melhor_divisao = dividir_dados(X, y, atributo, valor) # guardar os valores divididos

    # se foi encontrado um ganho continuar dividindo
    if melhor_ganho > 0:
        true_X, true_y, false_X, false_y = melhor_divisao
        true_branch = construir_arvore(true_X, true_y) #chamada recusriva para a classificação true
        false_branch = construir_arvore(false_X, false_y) #chamada recusriva para a classificação false
        new_no = Node(atributo=melhor_atributo, value=melhor_valor,true_branch=true_branch, false_branch=false_branch) #criar novo nó com essas informações
        return new_no # retornar o nó criado

    # condição de parada da recursão
    # se não houver ganho retorna nó folha
    classe_majoritaria = Counter(y).most_common(1)[0][0] # vai pegar a classe majoritária (se tiver mais "Sim" do que "Não", então a resposta será "Sim")
    new_no = Node(resp=classe_majoritaria) #retornar nó folha com a classe majoritária
    return new_no



# classificação/predição
def classificar(arvore, amostra):
    # percorrer a árvore com a amostra, até encontrar uma folha
    if arvore.resp is not None: #indica que é folha
        return arvore.resp
    if amostra[arvore.atributo] <= arvore.value:
        return classificar(arvore.true_branch, amostra)
    else:
        return classificar(arvore.false_branch, amostra)



# impressão recursiva da árvore
def imprimir_arvore(no, nomes_atributos, espacamento=""):
    if no.resp is not None:
        print(espacamento + "Resposta ->", no.resp)
        return
    nome = nomes_atributos[no.atributo] if no.atributo is not None else "?"
    print(espacamento + f"[{nome} <= {no.value}]")
    print(espacamento + "--> True:")
    imprimir_arvore(no.true_branch, nomes_atributos, espacamento + "   ")
    print(espacamento + "--> False:")
    imprimir_arvore(no.false_branch, nomes_atributos, espacamento + "   ")



# função pra mostrar as regras da árvore
def mostrar_regras(no, nomes_atributos, regra_atual=""):
    if no.resp is not None:
        # nó folha → imprime a regra completa com a classe
        print(regra_atual + f" ENTÃO Classe: {no.resp}")
        return
    
    nome = nomes_atributos[no.atributo] if no.atributo is not None else "?"
    
    # caminho True (<= valor)
    nova_regra_true = regra_atual + f"SE {nome} <= {no.value} "
    mostrar_regras(no.true_branch, nomes_atributos, nova_regra_true)
    
    # caminho False (> valor)
    nova_regra_false = regra_atual + f"SE {nome} > {no.value} "
    mostrar_regras(no.false_branch, nomes_atributos, nova_regra_false)


# exemplo com o titanic discretizado

with open('./titanic_Discret.pkl', 'rb') as f:
    X_treino, X_teste, y_treino, y_teste = pickle.load(f)

# guardar os nomes dos atributos antes de usar o numpy
nomes_atributos = list(X_treino.columns)

# converter para numpy arrays
X_treino = np.array(X_treino)
X_teste = np.array(X_teste)
y_treino = np.array(y_treino)
y_teste = np.array(y_teste)

arvore = construir_arvore(X_treino, y_treino)

# prever no conjunto de teste
y_pred = []
for i in range(len(X_teste)):
    pred = classificar(arvore, X_teste[i])
    y_pred.append(pred)
    print("Amostra:", X_teste[i], "Real:", y_teste[i], "Previsto:", pred)

# converter lista de previsões para numpy array
y_pred = np.array(y_pred)



# métricas
acuracia = accuracy_score(y_teste, y_pred)
precisao = precision_score(y_teste, y_pred, average='macro')  # 'macro' todos tem o mesmo peso
recall = recall_score(y_teste, y_pred, average='macro')
f1 = f1_score(y_teste, y_pred, average='macro')

print(f"Acurácia: {acuracia:.4f}")
print(f"Precisão: {precisao:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")




# matriz de Confusão
cm = confusion_matrix(y_teste, y_pred, labels=np.unique(y_teste))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_teste))
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - ID3")
plt.show()


# imprimir a árvore
imprimir_arvore(arvore, nomes_atributos)


# imprimir as regras da árvore
print("\nRegras da árvore: ")
mostrar_regras(arvore, nomes_atributos)