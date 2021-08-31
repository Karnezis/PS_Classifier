from PIL import Image
from io import BytesIO
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import keras


def load_model():
    # Defina aqui o caminho para o modelo a ser carregado
    model_path = 'src/model/ResNet50Amiloidosis_5.h5'
    # Carrega os pesos do modelo para uma variável auxiliar
    load_aux = keras.models.load_model(model_path)
    # Instancia um modelo com os pesos da auxiliar e camada de saída indicada
    model = Model(inputs=load_aux.inputs, outputs=load_aux.outputs)
    return model


model = load_model()


def read_imagefile(file):
    image = Image.open(BytesIO(file))
    return image


def preprocess(image: Image.Image):
    # Redimensiona a imagem
    image = np.asarray(image.resize((299, 299)))[..., :3]
    # Transforma a imagem em array
    #input_arr = np.asfarray(image)
    # Coloca mais uma dimensão no array
    input_arr = np.expand_dims(image, axis=0)
    # Normaliza o array para a rede conforme sua necessidade
    pre_processed_array = tf.keras.applications.resnet50.preprocess_input(
        input_arr)
    return pre_processed_array


def predict(processed_array: np.ndarray):
    # Prevê o array com a rede instanciada
    predicted_rate = model.predict(processed_array)
    return predicted_rate
