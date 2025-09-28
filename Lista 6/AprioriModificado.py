import pandas as pd
from apyori import apriori

# Parâmetros
min_suporte = 0.3
min_confianca = 0.6

# Lendo a base
base = pd.read_csv('supermercado.csv', sep=';', encoding='utf-8', dtype=str)

# Normalizar colunas e valores
base.columns = base.columns.str.strip()


# Construir transações com presença e ausência
transacoes = []
for i in range(len(base)):
    linha = []
    for col in base.columns:
        if base.at[i, col] == 'sim':
            linha.append(col)           # presença
        else:
            linha.append("¬" + col)     # ausência
    transacoes.append(linha)

# Executar Apriori
saida = list(apriori(transacoes, min_support=min_suporte, min_confidence=min_confianca))

# Função auxiliar para traduzir itens em texto natural
def traduz_item(item):
    if item.startswith("¬"):
        return f"não leva {item[1:]}"
    else:
        return f"leva {item}"

# Imprimir regras em linguagem natural
print("\nRegras geradas:")
for resultado in saida:
    s = resultado.support
    for regra in resultado.ordered_statistics:
        a = list(regra.items_base)
        b = list(regra.items_add)
        if not a or not b:
            continue

        antecedente_txt = " e ".join([traduz_item(x) for x in a])
        consequente_txt = " e ".join([traduz_item(x) for x in b])

        print(f"Quem {antecedente_txt} -> {consequente_txt} "
              f"(suporte={s:.2f}, confiança={regra.confidence:.2f}, lift={regra.lift:.2f})")