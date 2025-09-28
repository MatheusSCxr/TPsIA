import numpy as np
import pickle
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# classe nó da árvore
class Node:
    def __init__(self, atributo=None, value=None, resp=None, true_branch=None, false_branch=None):
        self.atributo = atributo         # índice do atributo usado para fazer a divisão nesse nó
        self.value = value               # valor de corte para o atributo
        self.resp = resp                 # resposta/classificação do nó folha
        self.true_branch = true_branch   # subárvore para <= valor
        self.false_branch = false_branch # subárvore para > valor


# funções auxiliares
def tratar_nulos(X):
    X = X.copy().astype(float)  # cria uma cópia da matriz e garante que os valores sejam float
    linhas, colunas = X.shape
    for j in range(colunas):
        soma, cont = 0.0, 0
        # percorre todas os valores
        for i in range(linhas):
            if not np.isnan(X[i][j]):  # se não for nulo
                soma += X[i][j]       # soma os valores
                cont += 1             # conta quantos valores válidos existem
        media = soma / cont if cont > 0 else 0.0  # calcula a média da coluna (se não tiver valores válidos, usa 0)
        for i in range(linhas):
            if np.isnan(X[i][j]):     # substitui valores nulos pela média da coluna
                X[i][j] = media
    return X


def dividir_dados(X, y, atributo, corte):
    true_indices, false_indices = [], []
    for i in range(len(X)):
        if X[i][atributo] <= corte:   # se o valor do atributo for menor ou igual ao corte
            true_indices.append(i)    # vai para o ramo verdadeiro
        else:
            false_indices.append(i)   # caso contrário, vai para o ramo falso
    return X[true_indices], y[true_indices], X[false_indices], y[false_indices]


def calcular_gini(y):
    _, counts = np.unique(y, return_counts=True)  # conta quantas vezes cada classe aparece
    probs = counts / len(y)                       # calcula a proporção de cada classe
    return 1.0 - np.sum(probs ** 2)               # fórmula do índice de gini


def ganho_gini(X, y, atributo, corte):
    _, true_y, _, false_y = dividir_dados(X, y, atributo, corte)  # divide os dados pelo corte
    p_true = len(true_y) / len(y) if len(y) > 0 else 0            # proporção do ramo verdadeiro
    p_false = len(false_y) / len(y) if len(y) > 0 else 0          # proporção do ramo falso
    gini_total = calcular_gini(y)                                 # gini do conjunto original
    gini_split = p_true * calcular_gini(true_y) + p_false * calcular_gini(false_y)
    return gini_total - gini_split                                # ganho de gini


# construção recursiva da árvore CART
def construir_arvore_CART(X, y):
    X = tratar_nulos(X)  # primeiro trata valores nulos substituindo pela média

    # caso base: todas as classes iguais
    if len(set(y)) == 1:
        return Node(resp=y[0])  # retorna nó folha com a classe única

    # variáveis "globais" que vão ser atualizadas de acordo com a construção da árvore
    melhor_ganho, melhor_atributo, melhor_valor, melhor_divisao = 0, None, None, None
    n_atributos = X.shape[1]  # número de atributos

    # percorre todos os atributos
    for atributo in range(n_atributos):
        valores_unicos = np.unique(X[:, atributo])  # pega todos os valores únicos do atributo
        for valor in valores_unicos:
            ganho = ganho_gini(X, y, atributo, valor)  # calcula o ganho de gini para esse valor
            if ganho > melhor_ganho:
                # se encontrar ganho melhor até agora então atualizar os valores globais
                melhor_ganho = ganho       # guarda o ganho
                melhor_atributo = atributo # guarda o índice do atributo
                melhor_valor = valor       # guarda o valor de corte
                melhor_divisao = dividir_dados(X, y, atributo, valor) # guarda a divisão feita

    # se foi encontrado um ganho continuar dividindo
    if melhor_ganho > 0:
        true_X, true_y, false_X, false_y = melhor_divisao
        true_branch = construir_arvore_CART(true_X, true_y)    # chamada recursiva para o ramo verdadeiro
        false_branch = construir_arvore_CART(false_X, false_y) # chamada recursiva para o ramo falso
        return Node(atributo=melhor_atributo, value=melhor_valor,
                    true_branch=true_branch, false_branch=false_branch)

    # condição de parada da recursão
    # se não houver ganho retorna nó folha
    classe_majoritaria = Counter(y).most_common(1)[0][0]  # pega a classe majoritária
    return Node(resp=classe_majoritaria)                  # retorna nó folha com a classe majoritária




# classificação/predição
def classificar(arvore, amostra):
    if arvore.resp is not None:
        return arvore.resp
    if amostra[arvore.atributo] <= arvore.value:
        return classificar(arvore.true_branch, amostra)
    else:
        return classificar(arvore.false_branch, amostra)

# ---------------------------
# Impressão da árvore (com nomes de atributos)
# ---------------------------
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

# ---------------------------
# Função para mostrar as regras da árvore
# ---------------------------
def mostrar_regras(no, nomes_atributos, regra_atual=""):
    if no.resp is not None:
        # Nó folha → imprime a regra completa com a classe
        print(regra_atual + f" ENTÃO Classe: {no.resp}")
        return
    
    nome = nomes_atributos[no.atributo] if no.atributo is not None else "?"
    
    # Caminho True (<= valor)
    nova_regra_true = regra_atual + f"SE {nome} <= {no.value} "
    mostrar_regras(no.true_branch, nomes_atributos, nova_regra_true)
    
    # Caminho False (> valor)
    nova_regra_false = regra_atual + f"SE {nome} > {no.value} "
    mostrar_regras(no.false_branch, nomes_atributos, nova_regra_false)





# exemplo com o tinanic com valores contínuos e missing
with open('./titanic_Continuo.pkl', 'rb') as f:
    X_treino, X_teste, y_treino, y_teste = pickle.load(f)

# guardar os nomes dos atributos antes de usar o numpy
nomes_atributos = list(X_treino.columns)

# converter para numpy
X_treino_np = X_treino.values
X_teste_np = X_teste.values
y_treino = np.array(y_treino)
y_teste = np.array(y_teste)

# construir árvore
arvore = construir_arvore_CART(X_treino_np, y_treino)

# prever no conjunto de teste
y_pred = []
for i in range(len(X_teste_np)):
    pred = classificar(arvore, X_teste_np[i])
    y_pred.append(pred)
    print("Amostra:", X_teste_np[i], "Real:", y_teste[i], "Previsto:", pred)

y_pred = np.array(y_pred)

# métricas
acuracia = accuracy_score(y_teste, y_pred)
precisao = precision_score(y_teste, y_pred, average='macro')
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
plt.title("Matriz de Confusão - CART")
plt.show()

# impressão da árvore com nomes
imprimir_arvore(arvore, nomes_atributos)

# imprimir as regras da árvore
print("\nRegras da árvore: ")
mostrar_regras(arvore, nomes_atributos)