import numpy as np
import pickle
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# classe Nó
class Node:
    def __init__(self, atributo=None, value=None, resp=None, branches=None, true_branch=None, false_branch=None, tipo=None):
        self.atributo = atributo      # índice do atributo
        self.value = value            # limiar (contínuo) ou dict de ramos (categórico)
        self.resp = resp              # classe se for folha
        self.branches = branches      # dict {categoria: Node} se categórico
        self.true_branch = true_branch  # nó filho (<= limiar)
        self.false_branch = false_branch # nó filho (> limiar)
        self.tipo = tipo              # "cont" ou "cat" (contínu ou categórico)


# funções auxiliares
def calcular_entropia(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum([p*np.log2(p) for p in probs if p > 0])

#tratar nulos numéricos e categóricos
def tratar_nulos(X):
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]

        # se for numérico → substitui NaN pela média
        if np.issubdtype(col.dtype, np.number):
            col = col.astype(float)
            mask = ~np.isnan(col)
            media = np.mean(col[mask]) if np.any(mask) else 0.0
            col[~mask] = media
            X[:, j] = col

        else:
            # se for categórico → substitui None ou np.nan pelo valor mais frequente
            col = col.astype(object)  # garante que tudo é tratado como objeto
            # filtra apenas valores válidos (não None, não nan)
            validos = [v for v in col if v is not None and v == v and v != ""]
            if len(validos) > 0:
                valores, counts = np.unique(validos, return_counts=True)
                mais_freq = valores[np.argmax(counts)]
                col = np.array([mais_freq if (v is None or v != v or v == "") else v for v in col], dtype=object)
                X[:, j] = col
    return X

# dividir dados para atributos contínuos
def dividir_dados_continuo(X, y, atributo, limiar):
    left_idx = X[:, atributo] <= limiar # índices onde o valor do atributo é menor ou igual ao limiar
    right_idx = X[:, atributo] > limiar # índices onde o valor do atributo é maior que o limiar
    return X[left_idx], y[left_idx], X[right_idx], y[right_idx] # retorna subconjuntos divididos


# dividir dados para atributos categóricos
def dividir_dados_categorico(X, y, atributo):
    valores = np.unique(X[:, atributo]) # obter valores únicos do atributo
    subsets = {} # dicionário para guardar subconjuntos
    for v in valores:
        idx = X[:, atributo] == v # índices onde o atributo é igual ao valor atual
        subsets[v] = (X[idx], y[idx]) # guardar subconjunto correspondente
    return subsets # retorna todos os subconjuntos


# calcular razão de ganho para atributos contínuos
def calcular_razaoGanho_continuo(X, y, atributo, limiar):
    X_left, y_left, X_right, y_right = dividir_dados_continuo(X, y, atributo, limiar) # dividir dados
    total = len(y) # número total de exemplos
    if len(y_left)==0 or len(y_right)==0: # se algum lado ficar vazio não há divisão válida
        return 0
    entropia_total = calcular_entropia(y) # entropia do conjunto original
    # entropia condicional ponderada pelos tamanhos dos subconjuntos
    entropia_cond = (len(y_left)/total)*calcular_entropia(y_left) + (len(y_right)/total)*calcular_entropia(y_right)
    ganho = entropia_total - entropia_cond # ganho de informação
    split_info = 0
    for subset in [y_left, y_right]:
        p = len(subset)/total # proporção de exemplos em cada subconjunto
        if p>0:
            split_info -= p*np.log2(p) # cálculo do split info
    return ganho/split_info if split_info>0 else 0 # retorna razão de ganho


# calcular razão de ganho para atributos categóricos
def calcular_razaoGanho_categorico(X, y, atributo):
    subsets = dividir_dados_categorico(X, y, atributo) # dividir dados em subconjuntos
    total = len(y) # número total de exemplos
    entropia_total = calcular_entropia(y) # entropia do conjunto original
    entropia_cond = 0
    split_info = 0
    for _, y_sub in subsets.values():
        p = len(y_sub)/total # proporção de exemplos no subconjunto
        entropia_cond += p*calcular_entropia(y_sub) # entropia condicional
        if p>0:
            split_info -= p*np.log2(p) # cálculo do split info
    ganho = entropia_total - entropia_cond # ganho de informação
    return ganho/split_info if split_info>0 else 0, subsets # retorna razão de ganho e subconjuntos


# detectar tipos de atributos automaticamente
def detectar_tipos(X, limite_cat=10):
    tipos = [] # lista para guardar o tipo de cada atributo
    for j in range(X.shape[1]):
        col = X[:, j] # coluna atual
        if np.issubdtype(col.dtype, np.number): # se for numérico
            if len(np.unique(col)) <= limite_cat: # se tiver poucos valores distintos
                tipos.append("cat") # trata como categórico
            else:
                tipos.append("cont") # trata como contínuo
        else:
            tipos.append("cat") # se não for numérico, é categórico
    return tipos # retorna lista com tipos de cada atributo


# construção recursiva da árvore
def construir_arvore(X, y, nomes_atributos, limite_cat=10):
    X = tratar_nulos(X) # tratar valores nulos antes de prosseguir

    # caso base: todas as classes iguais
    if len(set(y)) == 1:
        return Node(resp=y[0]) # retorna nó folha com a classe única

    # detectar se cada atributo é categórico ou contínuo
    tipos_atributos = detectar_tipos(X, limite_cat)

    # variáveis "globais" que vão ser atualizadas de acordo com a construção da árvore
    melhor_gain = 0
    melhor_atributo = None
    melhor_split_info = None
    tipo_split = None

    # percorrer todos os atributos
    for atributo in range(X.shape[1]):
        if tipos_atributos[atributo] == "cont":
            # ordenar valores únicos do atributo contínuo
            valores = np.unique(X[:, atributo])
            valores.sort()
            # testar limiares entre pares de valores consecutivos
            for i in range(len(valores)-1):
                limiar = (valores[i] + valores[i+1]) / 2
                gr = calcular_razaoGanho_continuo(X, y, atributo, limiar) # calcular razão de ganho
                if gr > melhor_gain:
                    # atualizar se encontrar melhor razão de ganho
                    melhor_gain = gr
                    melhor_atributo = atributo
                    melhor_split_info = limiar # guardar limiar de corte
                    tipo_split = "cont"
        else:
            # calcular razão de ganho para atributos categóricos
            gr, subsets = calcular_razaoGanho_categorico(X, y, atributo)
            if gr > melhor_gain:
                melhor_gain = gr
                melhor_atributo = atributo
                melhor_split_info = subsets # guardar subconjuntos resultantes
                tipo_split = "cat"

    # condição de parada: se não houver ganho retorna nó folha com classe majoritária
    if melhor_gain == 0:
        return Node(resp=Counter(y).most_common(1)[0][0])

    # se o melhor atributo for contínuo
    if tipo_split == "cont":
        limiar = melhor_split_info
        X_left, y_left, X_right, y_right = dividir_dados_continuo(X, y, melhor_atributo, limiar)
        return Node(
            atributo=melhor_atributo,
            value=limiar, # valor de corte
            tipo="cont", # marca como contínuo
            true_branch=construir_arvore(X_left, y_left, nomes_atributos, limite_cat), # chamada recursiva para <= limiar
            false_branch=construir_arvore(X_right, y_right, nomes_atributos, limite_cat) # chamada recursiva para > limiar
        )
    else:
        # se o melhor atributo for categórico
        branches = {}
        for valor, (X_sub, y_sub) in melhor_split_info.items():
            branches[valor] = construir_arvore(X_sub, y_sub, nomes_atributos, limite_cat) # chamada recursiva para cada categoria
        return Node(atributo=melhor_atributo, branches=branches, tipo="cat") # criar nó com múltiplos ramos
    


# classificação/predição com nós do tipo continuo e categórico
def classificar(no, amostra):
    if no.resp is not None:
        return no.resp
    if no.tipo == "cont":
        if amostra[no.atributo] <= no.value:
            return classificar(no.true_branch, amostra)
        else:
            return classificar(no.false_branch, amostra)
    else:
        valor = amostra[no.atributo]
        if valor in no.branches:
            return classificar(no.branches[valor], amostra)
        else:
            #retorna a classe majoritária do nó
            classes = []
            for branch in no.branches.values():
                c = classificar(branch, amostra)
                classes.append(c)

            contagem = Counter(classes)
            mais_comum = contagem.most_common(1)[0][0]
            return mais_comum



# impressão recursiva da árvore
def imprimir_arvore(no, nomes_atributos, espacamento=""):
    if no.resp is not None:
        print(espacamento + "Resposta ->", no.resp)
        return
    nome = nomes_atributos[no.atributo]
    if no.tipo == "cont":
        print(espacamento + f"[{nome} <= {no.value}]")
        print(espacamento + "--> True:")
        imprimir_arvore(no.true_branch, nomes_atributos, espacamento + "   ")
        print(espacamento + "--> False:")
        imprimir_arvore(no.false_branch, nomes_atributos, espacamento + "   ")
    else:
        for valor, branch in no.branches.items():
            print(espacamento + f"[{nome} == {valor}]")
            imprimir_arvore(branch, nomes_atributos, espacamento + "   ")



# função pra mostrar as regras da árvore
def mostrar_regras(no, nomes_atributos, regra_atual=""):
    if no.resp is not None:
        print(regra_atual + f" ENTÃO Classe: {no.resp}")
        return
    nome = nomes_atributos[no.atributo]
    if no.tipo == "cont":
        mostrar_regras(no.true_branch, nomes_atributos, regra_atual + f"SE {nome} <= {no.value} ")
        mostrar_regras(no.false_branch, nomes_atributos, regra_atual + f"SE {nome} > {no.value} ")
    else:
        for valor, branch in no.branches.items():
            mostrar_regras(branch, nomes_atributos, regra_atual + f"SE {nome} == {valor} ")



# exemplo com o tinanic com valores contínuos e missing
with open('./titanic_Continuo.pkl', 'rb') as f:   # ou './titanic.pkl'
    X_treino, X_teste, y_treino, y_teste = pickle.load(f)

# nomes dos atributos
nomes_atributos = list(X_treino.columns)

# converter para numpy
X_treino = np.array(X_treino)
X_teste = np.array(X_teste)
y_treino = np.array(y_treino)
y_teste = np.array(y_teste)

# construir árvore com detecção automática de tipos
arvore = construir_arvore(X_treino, y_treino, nomes_atributos)

# prever no conjunto de teste
y_pred = []
for i in range(len(X_teste)):
    pred = classificar(arvore, X_teste[i])
    y_pred.append(pred)
    print("Amostra:", X_teste[i], "Real:", y_teste[i], "Previsto:", pred)

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

# matriz de confusão
cm = confusion_matrix(y_teste, y_pred, labels=np.unique(y_teste))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_teste))
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão - C4.5")
plt.show()


# imprimir a árvore
imprimir_arvore(arvore, nomes_atributos)


# imprimir as regras
print("\nRegras da árvore: ")
mostrar_regras(arvore, nomes_atributos)