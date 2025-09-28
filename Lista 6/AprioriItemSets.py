import pandas as pd
from apyori import apriori

min_suporte = 0.3
min_confianca = 0.8

# Lendo a base
base = pd.read_csv('supermercado.csv', sep=';', encoding='utf-8')

# Transformando em lista de transações
transacoes = []
for i in range(len(base)):
    linha = []
    for j in range(len(base.columns)):
        if base.values[i, j] == 'Sim':   # só adiciona os itens comprados
            linha.append(base.columns[j])
    transacoes.append(linha)

# Executando Apriori
regras = apriori(transacoes, min_support=min_suporte, min_confidence=min_confianca)
saida = list(regras)

# Calculando o ItemSet1 manualmente
print("\nItemSet 1 (todos os itens)")
for col in base.columns:
    suporte = (base[col] == 'Sim').sum() / len(base)
    if (suporte >= min_suporte):
        print(f"  [{col}] -> suporte: {suporte:.2f}")

# Agrupar itemsets frequentes por tamanho (k = len(itemset))
itemsets_por_k = {}

for i in range(len(saida)):
    resultado = saida[i]
    itens = list(resultado.items)
    itens.sort()
    k = len(itens)
    if k not in itemsets_por_k:
        itemsets_por_k[k] = []
    itemsets_por_k[k].append((itens, resultado.support))

# Imprimir por tamanho do itemset (estilo mais "C-like")
chaves = list(itemsets_por_k.keys())
chaves.sort()

for idx in range(len(chaves)):
    k = chaves[idx]
    print("\nItemSet %d" % k)

    lista = itemsets_por_k[k]
    # ordenar manualmente por suporte decrescente
    lista_ordenada = sorted(lista, key=lambda x: x[1])

    for j in range(len(lista_ordenada)):
        itens, suporte = lista_ordenada[j]
        # imprime como se fosse printf em C
        print("  [", end="")
        for m in range(len(itens)):
            print(itens[m], end="")
            if m < len(itens) - 1:
                print(", ", end="")
        print("] -> suporte: %.2f" % suporte)




print()

# Extraindo regras
Antecedente = []
Consequente = []
suporte = []
confianca = []
lift = []

for resultado in saida:
    s = resultado.support
    for regra in resultado.ordered_statistics:
        a = list(regra.items_base)
        b = list(regra.items_add)
        c = regra.confidence
        l = regra.lift
        if len(a) == 0 or len(b) == 0: 
            continue
        Antecedente.append(a)
        Consequente.append(b)
        suporte.append(s)
        confianca.append(c)
        lift.append(l)

RegrasFinais = pd.DataFrame({
    'Antecedente': Antecedente,
    'Consequente': Consequente,
    'suporte': suporte,
    'confianca': confianca,
    'lift': lift
})

# Ordenando pelas regras mais fortes
print(RegrasFinais.sort_values(by='lift', ascending=False))