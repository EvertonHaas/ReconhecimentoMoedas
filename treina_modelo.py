from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np

def treinar_modelo():
    altura_imagem = 224  # Altura das imagens para o modelo
    largura_imagem = 224  # Largura das imagens para o modelo
    num_classes = 5  # Número de classes no seu dataset
    batch_size = 8  # Número de amostras processadas antes da atualização do modelo
    num_epochs = 80  # Número de passagens completas pelo dataset

    # Caminhos
    diretorio_treino = 'fotos/data/treino'  # Caminho para o diretório de treinamento

    # Número de Amostras
    # Você pode definir isso manualmente ou escrever um código para contar automaticamente
    numero_de_amostras_treino = 115  # Substitua pelo número real de imagens de treino

    def custom_normalization(img):
        return (img.astype(np.float32) / 127.0) - 1

    train_datagen = ImageDataGenerator(
        preprocessing_function=custom_normalization
        # outras configurações...
    )

    train_generator = train_datagen.flow_from_directory(
        diretorio_treino,
        target_size=(altura_imagem, largura_imagem),
        batch_size=batch_size,
        class_mode='categorical')

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(altura_imagem, largura_imagem, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))  # 'num_classes' é o número de classes
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=numero_de_amostras_treino // batch_size,
        epochs=num_epochs)

    model.save('modelo_treinado.keras')
