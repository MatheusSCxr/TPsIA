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
        self.tipo = tipo              # "cont" ou "cat" (contínuo ou categórico)


class C45:
    def __init__(self):
        self.arvore = None
        self.nomes_atributos = None

    # funções auxiliares
    def calcular_entropia(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum([p*np.log2(p) for p in probs if p > 0])

    # tratar nulos numéricos e categóricos
    def tratar_nulos(self, X):
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
                col = col.astype(object)
                validos = [v for v in col if v is not None and v == v and v != ""]
                if len(validos) > 0:
                    valores, counts = np.unique(validos, return_counts=True)
                    mais_freq = valores[np.argmax(counts)]
                    col = np.array([mais_freq if (v is None or v != v or v == "") else v for v in col], dtype=object)
                    X[:, j] = col
        return X

    # dividir dados para atributos contínuos
    def dividir_dados_continuo(self, X, y, atributo, limiar):
        left_idx = X[:, atributo] <= limiar # índices onde o valor do atributo é menor ou igual ao limiar
        right_idx = X[:, atributo] > limiar # índices onde o valor do atributo é maior que o limiar
        return X[left_idx], y[left_idx], X[right_idx], y[right_idx] # retorna subconjuntos divididos

    # dividir dados para atributos categóricos
    def dividir_dados_categorico(self, X, y, atributo):
        valores = np.unique(X[:, atributo]) # obter valores únicos do atributo
        subsets = {} # dicionário para guardar subconjuntos
        for v in valores:
            idx = X[:, atributo] == v # índices onde o atributo é igual ao valor atual
            subsets[v] = (X[idx], y[idx]) # guardar subconjunto correspondente
        return subsets # retorna todos os subconjuntos

    # calcular razão de ganho para atributos contínuos
    def calcular_razaoGanho_continuo(self, X, y, atributo, limiar):
        X_left, y_left, X_right, y_right = self.dividir_dados_continuo(X, y, atributo, limiar) # dividir dados
        total = len(y) # número total de exemplos
        if len(y_left)==0 or len(y_right)==0: # se algum lado ficar vazio não há divisão válida
            return 0
        entropia_total = self.calcular_entropia(y) # entropia do conjunto original
        entropia_cond = (len(y_left)/total)*self.calcular_entropia(y_left) + (len(y_right)/total)*self.calcular_entropia(y_right)
        ganho = entropia_total - entropia_cond # ganho de informação
        split_info = 0
        for subset in [y_left, y_right]:
            p = len(subset)/total
            if p>0:
                split_info -= p*np.log2(p)
        return ganho/split_info if split_info>0 else 0

    # calcular razão de ganho para atributos categóricos
    def calcular_razaoGanho_categorico(self, X, y, atributo):
        subsets = self.dividir_dados_categorico(X, y, atributo)
        total = len(y)
        entropia_total = self.calcular_entropia(y)
        entropia_cond = 0
        split_info = 0
        for _, y_sub in subsets.values():
            p = len(y_sub)/total
            entropia_cond += p*self.calcular_entropia(y_sub)
            if p>0:
                split_info -= p*np.log2(p)
        ganho = entropia_total - entropia_cond
        return ganho/split_info if split_info>0 else 0, subsets

    # detectar tipos de atributos automaticamente
    def detectar_tipos(self, X, limite_cat=10):
        tipos = []
        for j in range(X.shape[1]):
            col = X[:, j]
            if np.issubdtype(col.dtype, np.number):
                if len(np.unique(col)) <= limite_cat:
                    tipos.append("cat")
                else:
                    tipos.append("cont")
            else:
                tipos.append("cat")
        return tipos

    # construção recursiva da árvore
    def construir_arvore(self, X, y, nomes_atributos, limite_cat=10):
        X = self.tratar_nulos(X)

        # caso base: todas as classes iguais
        if len(set(y)) == 1:
            return Node(resp=y[0])

        tipos_atributos = self.detectar_tipos(X, limite_cat)

        melhor_gain = 0
        melhor_atributo = None
        melhor_split_info = None
        tipo_split = None

        for atributo in range(X.shape[1]):
            if tipos_atributos[atributo] == "cont":
                valores = np.unique(X[:, atributo])
                valores.sort()
                for i in range(len(valores)-1):
                    limiar = (valores[i] + valores[i+1]) / 2
                    gr = self.calcular_razaoGanho_continuo(X, y, atributo, limiar)
                    if gr > melhor_gain:
                        melhor_gain = gr
                        melhor_atributo = atributo
                        melhor_split_info = limiar
                        tipo_split = "cont"
            else:
                gr, subsets = self.calcular_razaoGanho_categorico(X, y, atributo)
                if gr > melhor_gain:
                    melhor_gain = gr
                    melhor_atributo = atributo
                    melhor_split_info = subsets
                    tipo_split = "cat"

        if melhor_gain == 0:
            return Node(resp=Counter(y).most_common(1)[0][0])

        if tipo_split == "cont":
            limiar = melhor_split_info
            X_left, y_left, X_right, y_right = self.dividir_dados_continuo(X, y, melhor_atributo, limiar)
            return Node(
                atributo=melhor_atributo,
                value=limiar,
                tipo="cont",
                true_branch=self.construir_arvore(X_left, y_left, nomes_atributos, limite_cat),
                false_branch=self.construir_arvore(X_right, y_right, nomes_atributos, limite_cat)
            )
        else:
            branches = {}
            for valor, (X_sub, y_sub) in melhor_split_info.items():
                branches[valor] = self.construir_arvore(X_sub, y_sub, nomes_atributos, limite_cat)
            return Node(atributo=melhor_atributo, branches=branches, tipo="cat")

    # classificação/predição
    def classificar(self, no, amostra):
        if no.resp is not None:
            return no.resp
        if no.tipo == "cont":
            if amostra[no.atributo] <= no.value:
                return self.classificar(no.true_branch, amostra)
            else:
                return self.classificar(no.false_branch, amostra)
        else:
            valor = amostra[no.atributo]
            if valor in no.branches:
                return self.classificar(no.branches[valor], amostra)
            else:
                classes = []
                for branch in no.branches.values():
                    c = self.classificar(branch, amostra)
                    classes.append(c)
                contagem = Counter(classes)
                return contagem.most_common(1)[0][0]

    # impressão recursiva da árvore
    def imprimir_arvore(self, no, nomes_atributos, espacamento=""):
        if no.resp is not None:
            print(espacamento + "Resposta ->", no.resp)
            return
        nome = nomes_atributos[no.atributo]
        if no.tipo == "cont":
            print(espacamento + f"[{nome} <= {no.value}]")
            print(espacamento + "--> True:")
            self.imprimir_arvore(no.true_branch, nomes_atributos, espacamento + "   ")
            print(espacamento + "--> False:")
            self.imprimir_arvore(no.false_branch, nomes_atributos, espacamento + "   ")
        else:
            for valor, branch in no.branches.items():
                print(espacamento + f"[{nome} == {valor}]")
                self.imprimir_arvore(branch, nomes_atributos, espacamento + "   ")

    # função pra mostrar as regras da árvore
    def mostrar_regras(self, no, nomes_atributos, regra_atual=""):
        if no.resp is not None:
            print(regra_atual + f" ENTÃO Classe: {no.resp}")
            return
        nome = nomes_atributos[no.atributo]
        if no.tipo == "cont":
            self.mostrar_regras(no.true_branch, nomes_atributos, regra_atual + f"SE {nome} <= {no.value} ")
            self.mostrar_regras(no.false_branch, nomes_atributos, regra_atual + f"SE {nome} > {no.value} ")
        else:
            for valor, branch in no.branches.items():
                self.mostrar_regras(branch, nomes_atributos, regra_atual + f"SE {nome} == {valor} ")

    # método para treinar (fit)
    def fit(self, X, y, nomes_atributos=None):
        self.nomes_atributos = nomes_atributos
        self.arvore = self.construir_arvore(np.array(X), np.array(y), nomes_atributos)

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

        # matriz de confusão
        cm = confusion_matrix(y, y_pred, labels=np.unique(y))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Matriz de Confusão - C4.5")
        plt.show()

    # salvar modelo
    def salvar(self, caminho="c45.pkl"):
        with open(caminho, "wb") as f:
            pickle.dump(self.arvore, f)

    # carregar modelo
    def carregar(self, caminho="c45.pkl"):
        with open(caminho, "rb") as f:
            self.arvore = pickle.load(f)