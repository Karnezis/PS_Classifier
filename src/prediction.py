from PIL import Image
from io import BytesIO
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import keras
from keras.models import model_from_json
from gradcam.visualizer import Visualizer


def load_models():
    # ------------- Amiloidose ------------------------------------------------
    # Defina aqui o caminho para o modelo a ser carregado
    amiloidosis_model_path = 'src/model/ResNet50Amiloidosis_5.h5'
    # Carrega os pesos do modelo para uma variável auxiliar
    load_aux = keras.models.load_model(amiloidosis_model_path)
    # Instancia um modelo com os pesos da auxiliar e camada de saída indicada
    amiloidosis_model = Model(inputs=load_aux.inputs, outputs=load_aux.outputs)
    # ------------- Esclerose -------------------------------------------------
    arquivo = open('src/model/sclerosis-model.json', 'r')
    estrutura_rede = arquivo.read()
    arquivo.close()
    sclerosis_model = model_from_json(estrutura_rede)
    sclerosis_model.load_weights('src/model/sclerosis-weights.h5')
    # ------------- Hipercelularidade -----------------------------------------
    hiper_model = keras.models.load_model(
        'src/model/hiper-bestModel', compile=False)
    hiper_model.load_weights('src/model/hiper-bestModel')
    return amiloidosis_model, sclerosis_model, hiper_model


amiloidosis_model, sclerosis_model, hiper_model = load_models()


def read_imagefile(file):
    image = Image.open(BytesIO(file))
    return image


def preprocess_amiloidosis(image: Image.Image):
    # Redimensiona a imagem
    image = np.asarray(image.resize((299, 299)))[..., :3]
    # Coloca mais uma dimensão no array
    input_arr = np.expand_dims(image, axis=0)
    # Normaliza o array para a rede conforme sua necessidade
    pre_processed_array = tf.keras.applications.resnet50.preprocess_input(
        input_arr)
    return pre_processed_array


def preprocess_sclerosis(image: Image.Image):
    # Redimensiona a imagem
    image = np.asarray(image.resize((224, 224)))[..., :3]
    # Normaliza o array para a rede conforme sua necessidade
    image = image/255
    # Coloca mais uma dimensão no array
    pre_processed_array = np.expand_dims(image, axis=0)
    return pre_processed_array


def predict_amiloidosis(processed_array: np.ndarray):
    # Prevê o array com a rede instanciada
    predicted_rate = amiloidosis_model.predict(processed_array)
    return predicted_rate


def predict_sclerosis(processed_array: np.ndarray):
    # Prevê o array com a rede instanciada
    predicted_rate = sclerosis_model.predict(processed_array)
    return predicted_rate


def visualizer(self, imagePath, model):
    return self.gradcam.visualize(imagePath, model, 'middle', self.visLabel, 'CAM_IMAGE_JET', self.visGuided)
