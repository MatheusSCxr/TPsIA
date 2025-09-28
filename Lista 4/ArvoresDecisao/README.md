# Biblioteca de Árvores de Decisão

Esta biblioteca reúne três implementações recursivas de árvores de decisão: **ID3**, **C4.5** e **CART**, padronizadas em classes com métodos compatíveis (`fit`, `predict`, `avaliar`, `imprimir_arvore`, `mostrar_regras`, `salvar`, `carregar`).  

---

## 📂 Estrutura do Projeto
```
ArvoresDecisao/ │ ├── setup.py ├── README.md ├── requirements.txt └── ArvoresDecisao/ ├── init.py ├── ID3.py ├── C45.py ├── CART.py
```

## ⚙️ Instalação

1. Instale as dependências:

```bash
pip install -r requirements.txt
```

2. Instale a biblioteca localmente:
```bash
pip install -e .
```


## 🚀 Uso Rápido
Exemplo com ID3

```python
from ArvoresDecisao import ID3

modelo = ID3()
modelo.fit(X_treino, y_treino, nomes_atributos=list(X_treino.columns))
y_pred = modelo.predict(X_teste)

modelo.imprimir_arvore(modelo.arvore, modelo.nomes_atributos)
modelo.mostrar_regras(modelo.arvore, modelo.nomes_atributos)
modelo.avaliar(X_teste, y_teste)

modelo.salvar("meu_id3.pkl")
modelo.carregar("meu_id3.pkl")
```

## 📌 Observações
- ID3: funciona com atributos discretizados.
- C4.5: trata atributos contínuos, categóricos e valores faltantes.
- CART: usa índice de Gini e trata nulos numéricos substituindo pela média.
- Todos os algoritmos seguem a mesma API para facilitar comparação.
