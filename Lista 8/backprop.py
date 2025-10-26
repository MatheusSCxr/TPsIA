import numpy as np
import random

# Funções auxiliares

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(y):
    # y já é o valor do sigmoid(x)
    return y * (1 - y)

def one_hot(n, classes=10):
    v = np.zeros(classes)
    v[n] = 1
    return v

def to_binary4(n):
    # Converte número 0..9 em 4 bits (ex: 5 -> [0,1,0,1])
    bits = [(n >> i) & 1 for i in range(4)]
    return np.array(bits[::-1], dtype=float)  # MSB à esquerda



# Classe da rede neural do backpropagation
class BackPropagation:
    def __init__(self, n_in, n_oculta, n_out, ta=0.5):
        # Pesos aleatórios pequenos
        self.W1 = np.random.uniform(-1, 1, (n_oculta, n_in))
        self.b1 = np.random.uniform(-1, 1, n_oculta)
        self.W2 = np.random.uniform(-1, 1, (n_out, n_oculta))
        self.b2 = np.random.uniform(-1, 1, n_out)
        self.ta = ta

    def forward(self, x):
        # Camada oculta
        z1 = self.W1 @ x + self.b1
        a1 = sigmoid(z1)
        # Camada de saída
        z2 = self.W2 @ a1 + self.b2
        a2 = sigmoid(z2)
        return a1, a2

    def train_step(self, x, target, epoca=None):
        # Forward
        a1, y = self.forward(x)

        # Erro da saída
        delta2 = (y - target) * sigmoid_deriv(y)
        # Gradientes da saída
        dW2 = np.outer(delta2, a1)
        db2 = delta2

        # Erro da camada oculta
        delta1 = (self.W2.T @ delta2) * sigmoid_deriv(a1)

        dW1 = np.outer(delta1, x)
        db1 = delta1

        # Atualização dos pesos
        self.W2 -= self.ta * dW2
        self.b2 -= self.ta * db2
        self.W1 -= self.ta * dW1
        self.b1 -= self.ta * db1

        # Mostrar ajustes nos pesos a cada 5000 epocas
        if epoca is not None and epoca % 5000 == 0:
            print(f"\n=== Época {epoca} ===")
            print("Equação geral: W = W - η * ΔW")
            print(f"ΔW2 = {np.round(dW2, 4)}")
            print(f"Δb2 = {np.round(db2, 4)}")
            print(f"ΔW1 = {np.round(dW1, 4)}")
            print(f"Δb1 = {np.round(db1, 4)}")
            print(f"Novos W2 =\n{np.round(self.W2, 4)}")
            print(f"Novos b2 = {np.round(self.b2, 4)}")
            print(f"Novos W1 =\n{np.round(self.W1, 4)}")
            print(f"Novos b1 = {np.round(self.b1, 4)}")

    def predict_bits(self, x):
        _, y = self.forward(x)
        bits = (y >= 0.5).astype(int)
        return bits

    def predict_digit(self, x):
        bits = self.predict_bits(x)
        # Converte lista de bits em número inteiro
        value = int("".join(map(str, bits)), 2)
        if value < 10:
            return value, one_hot(value, 10)
        else:
            return None, None


# Treinando e testando 


# Dados do XOR

inputs = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
], dtype=float)

targets = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)



# Treinamento

net = BackPropagation(n_in=2, n_oculta=2, n_out=1, ta=0.5) # 2 entradas, 2 neuronios na camada oculta, 1 na camada de saída. ta = 0.5

# Treinando em 10.000 epocas
for epoca in range(10000):
    i = np.random.randint(0, 4)
    net.train_step(inputs[i], targets[i], epoca=epoca)

# Testes manuais
test_cases = [
    ([0,0], "0"),   # XOR correto
    ([0,1], "1"),
    ([1,0], "1"),
    ([1,1], "0"),
]

print("\n--- Testes XOR ---")
for arr, esperado in test_cases:
    entrada = np.array(arr, dtype=float)
    y = net.predict_bits(entrada)
    pred = int(y >= 0.5)  # threshold
    print(f"Entrada: {arr} | Esperado: {esperado} | Previsto: {pred} | Saída bruta: {np.round(y,3)}")



# Testes XOR com ruído
def add_noise_xor(x, noise_level=0.3):
    """Adiciona ruído gaussiano às entradas do XOR"""
    noisy = x + np.random.normal(0, noise_level, size=x.shape)
    return np.clip(noisy, 0, 1)  # mantém no intervalo [0,1]

print("\n--- Testes XOR com ruído ---")
acertos = 0
total = 1000
for _ in range(total):
    i = np.random.randint(0, 4)          # escolhe uma das 4 combinações do XOR
    entrada = inputs[i]                  # entrada original
    alvo = targets[i]                    # saída esperada
    ruidosa = add_noise_xor(entrada)     # aplica ruído gaussiano
    y_pred = net.predict_bits(ruidosa)   # previsão da rede
    pred = int(y_pred[0])                # converte para escalar
    if pred == int(alvo[0]):
        acertos += 1
    if _ < 10:  # mostra apenas alguns exemplos
        print(f"Entrada real: {entrada} | Ruidosa: {np.round(ruidosa,2)} | Previsto: {pred} | Esperado: {int(alvo[0])}")
print(f"\nAcurácia média com ruído XOR: {acertos/total:.2%}")


# Dados: dígitos em 7 segmentos
digits_7seg = {
    0: [1,1,1,1,1,1,0],
    1: [0,1,1,0,0,0,0],
    2: [1,1,0,1,1,0,1],
    3: [1,1,1,1,0,0,1],
    4: [0,1,1,0,0,1,1],
    5: [1,0,1,1,0,1,1],
    6: [1,0,1,1,1,1,1],
    7: [1,1,1,0,0,0,0],
    8: [1,1,1,1,1,1,1],
    9: [1,1,1,1,0,1,1],
}

inputs = np.array([digits_7seg[d] for d in range(10)], dtype=float)
targets = np.array([to_binary4(d) for d in range(10)], dtype=float)

# Treinamento
net2 = BackPropagation(n_in=7, n_oculta=5, n_out=4, ta=0.5) # 7 entradas, 5 neuronios na camada oculta, 4 na camada de saída. ta = 0.5

# Treinando em 20.000 epocas
for epoca in range(20000):
    i = np.random.randint(0, 10)
    net2.train_step(inputs[i], targets[i], epoca=epoca)


# Testes

# Conjunto de testes
test_cases = [
    ([0,1,1,0,0,0,0], "1"),   # dígito 1 correto
    ([1,1,0,1,1,0,1], "2"),   # dígito 2 correto
    ([1,1,1,1,0,0,1], "3"),   # dígito 3 correto
    ([0,1,1,0,0,1,1], "4"),   # dígito 4 correto
    ([1,0,1,1,0,1,1], "5"),   # dígito 5 correto
    ([1,0,1,1,1,1,1], "6"),   # dígito 6 correto
    ([1,1,1,0,0,0,0], "7"),   # dígito 7 correto
    ([1,1,1,1,1,1,1], "8"),   # dígito 8 correto
    ([1,1,1,1,0,1,1], "9"),   # dígito 9 correto
]

print("\n--- Testes 7 segmentos ---")
for arr, esperado in test_cases:
    entrada = np.array(arr, dtype=float)
    pred_val, pred_onehot = net2.predict_digit(entrada)
    print(f"Entrada: {arr} | Esperado: {esperado} | Previsto: {pred_val} | One-hot previsto: {pred_onehot}")


# Função para adicionar ruído n
def add_noise_7seg(x, prob=0.2):
    """Inverte cada segmento com probabilidade 'prob'."""
    noisy = x.copy()
    for i in range(len(noisy)):
        if random.random() < prob:
            noisy[i] = 1 - noisy[i]  # inverte 0 para 1 ou 1 para 0
    return noisy

# Testes com ruído
print("\n--- Testes com ruído nos dígitos 7 segmentos ---")
for d in range(10):
    entrada = np.array(digits_7seg[d], dtype=float)
    ruidosa = add_noise_7seg(entrada, prob=0.2)  # 20% de chance de falha em cada segmento
    pred_val, _ = net2.predict_digit(ruidosa)
    print(f"Dígito real: {d} | Entrada ruidosa: {ruidosa.astype(int)} | Previsto: {pred_val}")

print("\n--- Avaliação de robustez ---")
acertos = 0
total = 1000
for _ in range(total):
    d = np.random.randint(0, 10)  # escolhe um dígito aleatório
    entrada = np.array(digits_7seg[d], dtype=float)  # padrão correto
    ruidosa = add_noise_7seg(entrada, prob=0.2)      # aplica ruído (20% de chance de inverter cada segmento)
    pred_val, _ = net2.predict_digit(ruidosa)        # previsão da rede
    if pred_val == d:
        acertos += 1
print(f"Acurácia com ruído: {acertos/total:.2%}")
