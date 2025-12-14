import os
import random
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing import image


# CONFIGURAÇÕES GLOBAIS E CAMINHOS

# Caminho onde estão as pastas originais do dataset (PetImages)
DIRETORIO_ORIGINAL = './PetImages'

# Caminho onde será criada a base dividida (treino/teste)
BASE_DIR = './dataset_organizado' 

# Parâmetros
TAMANHO_IMAGEM = (150, 150)
BATCH_SIZE = 64 # Depois de testes, foi o melhor valor que encontrei para um tempo razoável de treino no meu computador
SPLIT_SIZE = 0.8  # 80% para Treino 20% para Teste

# Hiperparâmetros Ajustados
# Taxa de aprendizado reduzida para estabilizar o treinamento, conforme análise gráfica
LEARNING_RATE = 0.001  # Padrão do otimizador Adam
EPOCHS = 30 
PATIENCE = 5 # Limite de épocas antes da parada




# ORGANIZAÇÃO E SEPARAÇÃO

# Divisão do treino/teste sobre a pasta PetImages
def criar_e_organizar_pastas(diretorio_original, base_dir, split_size):
    """Cria a estrutura de pastas de treino/teste e move arquivos."""
    print("Iniciando organização e separação dos dados (80/20)...")

    CAT_SOURCE_DIR = os.path.join(diretorio_original, 'Cat')
    DOG_SOURCE_DIR = os.path.join(diretorio_original, 'Dog')

    if not os.path.isdir(CAT_SOURCE_DIR) or not os.path.isdir(DOG_SOURCE_DIR):
        print("\nERRO: Diretórios 'Cat' ou 'Dog' não encontrados. Verifique a base de dados.")
        return False

    # Limpeza e Criação de Pastas
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)

    for folder in ['treino', 'teste']:
        os.makedirs(os.path.join(base_dir, folder, 'cats'))
        os.makedirs(os.path.join(base_dir, folder, 'dogs'))

    # Separa as imagens de treino/teste
    def separar_arquivos(source_dir, dest_train_dir, dest_test_dir, split):
        """Move e separa os arquivos em treino e teste."""
        files = [f for f in os.listdir(source_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        # Remove arquivos corrompidos comuns na base do Kaggle (ex: 666.jpg ou 11702.jpg)
        files = [f for f in files if f not in ('666.jpg', '11702.jpg')]
        random.shuffle(files)
        
        split_point = int(len(files) * split)

        # Copia para Treino e Teste
        for file in files[:split_point]:
            shutil.copyfile(os.path.join(source_dir, file), os.path.join(dest_train_dir, file))
        for file in files[split_point:]:
            shutil.copyfile(os.path.join(source_dir, file), os.path.join(dest_test_dir, file))
        
        print(f"  - {len(files)} imagens de {os.path.basename(source_dir)} processadas e separadas.")

    # Separar de acordo com a classe
    separar_arquivos(CAT_SOURCE_DIR, 
                     os.path.join(base_dir, 'treino', 'cats'), 
                     os.path.join(base_dir, 'teste', 'cats'), 
                     split_size)

    # Separar de acordo com a classe
    separar_arquivos(DOG_SOURCE_DIR, 
                     os.path.join(base_dir, 'treino', 'dogs'), 
                     os.path.join(base_dir, 'teste', 'dogs'), 
                     split_size)

    print("Organização e separação concluídas.")
    return True

if not criar_e_organizar_pastas(DIRETORIO_ORIGINAL, BASE_DIR, SPLIT_SIZE):
    exit()




# PRÉ-PROCESSAMENTO E DATA AUGMENTATION

print("\nIniciando pré-processamento...")

# Gerador de TREINO (Com Normalização e Aumento de Dados)
train_datagen = ImageDataGenerator(
    rescale=1./255,             # Normalização de pixels (0-1)
    rotation_range=40,          # Rotação
    width_shift_range=0.2,      # Deslocamento Horizontal
    height_shift_range=0.2,     # Deslocamento Vertical
    shear_range=0.2,            # Cisalhamento
    zoom_range=0.2,             # Zoom
    horizontal_flip=True,       # Inversão Horizontal
    fill_mode='nearest'         # Preenchimento
)

# Gerador de TESTE (Apenas Normalização)
test_datagen = ImageDataGenerator(rescale=1./255)


# Carregamento dos dados
print("\nCarregando dados de TREINO...")
train_generator = train_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'treino'),
    target_size=TAMANHO_IMAGEM,
    batch_size=BATCH_SIZE,
    class_mode='binary',
)

print("Carregando dados de TESTE...")
validation_generator = test_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'teste'),
    target_size=TAMANHO_IMAGEM,
    batch_size=BATCH_SIZE,
    class_mode='binary',
)

# Mostrar classes identificadas
print(f"\nClasses encontradas: {train_generator.class_indices}")




# CONSTRUÇÃO E TREINAMENTO DA CNN

print("\nConstruindo o modelo CNN...")

# Arquitetura simples: Conv -> Pool -> Flatten -> Dense -> Output
model = Sequential([
    # Bloco 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),

    # Bloco 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Bloco 3
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Camadas de Classificação
    Flatten(),
    Dropout(0.3),# Regularização para evitar overfitting
    Dense(256, activation='relu'), # Era 512 -> 256 para melhorar a performance
    
    # Camada de Saída (Sigmoid para classificação binária)
    Dense(1, activation='sigmoid') 
])

# Compilação do modelo com o LEARNING_RATE de 0.001
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), # Usa o otimizador Adam com taxa de aprendizado definida
              loss='binary_crossentropy', # Função de perda adequada para classificação binária
              metrics=['accuracy']) # Mais adequada para avaliar o desempenho do modelo

model.summary()

# Definição do Early Stopping (parada antecipada)
early_stop = EarlyStopping(monitor='val_loss', # Monitora a perda de validação
                           patience=PATIENCE,   # Número de épocas sem melhoria para parar
                           restore_best_weights=True) # Volta aos pesos da melhor época

print("\nIniciando treinamento...")


# Treinar o modelo
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stop], # parada antecipada
)


# Plotagem dos Gráficos de Desempenho
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Acurácia de Treino')
plt.plot(epochs_range, val_acc, label='Acurácia de Validação')
plt.legend(loc='lower right')
plt.title('Acurácia de Treino e Validação')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Perda de Treino')
plt.plot(epochs_range, val_loss, label='Perda de Validação')
plt.legend(loc='upper right')
plt.title('Perda (Loss) de Treino e Validação')
plt.show()


# AVALIAÇÃO E TESTES

# AVALIAÇÃO DO MODELO (PRECISÃO, RECALL, F1-SCORE)

print("\n--- Avaliação de Desempenho no Conjunto de Teste ---")

# Gerador de avaliação com shuffle=False para garantir que a ordem dos labels seja correta
eval_datagen = ImageDataGenerator(rescale=1./255)

eval_generator = eval_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'teste'),
    target_size=TAMANHO_IMAGEM,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# 1. Fazer previsões
# steps garante que o modelo passe por TODAS as imagens do teste.
Y_pred = model.predict(eval_generator, steps=len(eval_generator)) 

# 2. Converter probabilidades para classes (0 ou 1)
y_pred_classes = np.where(Y_pred > 0.5, 1, 0) 

# 3. Obter classes verdadeiras (elas estarão na ordem correta devido ao shuffle=False)
y_true = eval_generator.classes

# 4. Gerar Relatório de Classificação e Matriz de Confusão
target_names = list(train_generator.class_indices.keys())

print("\nRelatório de Classificação (Precision, Recall, F1-Score):")
# Visualização correta das classes 0 e 1
print(classification_report(y_true, y_pred_classes, target_names=target_names))

print("\nMatriz de Confusão:")
print(confusion_matrix(y_true, y_pred_classes))


# Teste com Novas Imagens
NOVO_TESTE_DIR = './testes_externos' 

def predizer_nova_imagem(img_path, model, target_size, class_indices):
    """Carrega, preprocessa e prediz a classe de uma única imagem."""
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        
        classes = {v: k for k, v in class_indices.items()}
        
        if prediction[0] > 0.5:
            resultado = classes[1] # dogs (1)
            probabilidade = prediction[0][0]
        else:
            resultado = classes[0] # cats (0)
            probabilidade = 1.0 - prediction[0][0]

        print(f"Arquivo: {os.path.basename(img_path)}")
        print(f"  Previsão: {resultado} (Probabilidade: {probabilidade:.4f})")
        
        plt.figure()
        plt.imshow(img)
        plt.title(f"Previsão: {resultado} ({probabilidade*100:.2f}%)")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"Não foi possível processar a imagem {img_path}. Erro: {e}")
        
print("\n--- 3.2 Teste com Novas Imagens ---")

if os.path.isdir(NOVO_TESTE_DIR):
    new_images_to_test = [os.path.join(NOVO_TESTE_DIR, f) for f in os.listdir(NOVO_TESTE_DIR) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if new_images_to_test:
        for img_path in new_images_to_test:
            predizer_nova_imagem(img_path, model, TAMANHO_IMAGEM, train_generator.class_indices)
    else:
        print(f"Nenhuma imagem encontrada em {NOVO_TESTE_DIR}. Adicione as imagens para o teste final.")
else:
    print(f"Pasta de teste '{NOVO_TESTE_DIR}' não encontrada. Crie a pasta e adicione imagens externas.")
