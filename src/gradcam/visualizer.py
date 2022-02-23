# -*- coding: utf-8 -*-
from operator import mod
from matplotlib import pyplot as plt
from PIL import Image
import os
import numpy as np
from .gradcam import grad_cam
from .guided_gradcam import guided_grad_cam
from .deprocess import Method, create_cam_image, create_guided_cam_image, convert_to_bgr, plot
from .model import load_model, get_model_viewable_layers, get_model_nb_classes
from .util import save_model_summary, Cam, create_folder_if_not_exists, extract_file_name
from .image import load_image, preprocess_image, save_image
from tensorflow.keras.models import Model
import keras
from keras.models import model_from_json


class Visualizer:

    def __init__(self):
        self.layer_name = 'ALL'
        self.amiloidosis_model, self.sclerosis_model, self.hiper_model = self.load_models()

    def load_models(self):
        # ------------- Amiloidose ------------------------------------------------
        # Defina aqui o caminho para o modelo a ser carregado
        amiloidosis_model_path = 'src/model/ResNet50Amiloidosis_5.h5'
        # Carrega os pesos do modelo para uma variável auxiliar
        load_aux = keras.models.load_model(amiloidosis_model_path)
        # Instancia um modelo com os pesos da auxiliar e camada de saída indicada
        amiloidosis_model = Model(
            inputs=load_aux.inputs, outputs=load_aux.outputs)
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

    def load_image(self, image_path):
        image = Image.open(image_path, )  # Open Image
        image = image.convert('RGB')  # Convert to RGB
        image = np.asarray(image)  # Convert to numpy array
        return image

    def preprocess_image(self, image, size=(224, 224)):
        '''
        Preprocess Image resizing it to model's first layer shape
        and scale rgb values to [0,1]
        '''
        _, _, chan = image.shape
        assert chan == 3
        image = np.array(Image.fromarray(image).resize(size, resample=2))
        #image = scipy.misc.imresize(image, size, interp='bilinear')
        image = image/255.
        image = np.array(image, dtype='float32')
        image = np.expand_dims(image, axis=0)
        return image

    def get_model_viewable_layers(self, model):
        '''
        Get a list with model's viewable layers names

        '''
        return list(dict([(layer.name, layer) for layer in model.layers if len(layer.output_shape) == 4]).keys())

    def get_model_nb_classes(self, model):
        '''
        Get number of classes from a model

        '''
        return self.model.layers[-1].output_shape[1]

    def visualize(self, image_file, model_opt, layer_name, label, method_name, guided):
        self.image_file = image_file
        if (model_opt == 1):
            self.model = self.amiloidosis_model
        elif (model_opt == 2):
            self.model = self.sclerosis_model
        else:
            self.model = self.hiper_model
        # 2. Load image
        image = self.load_image(image_file)
        height, width, _ = image.shape
        # Get model's input shape
        _, input_width, input_height, _ = self.model.layers[0].input_shape
        # 2.1 Preprocess Image
        preprocessed_image = self.preprocess_image(
            image, (input_width, input_height))
        all_layers = self.get_model_viewable_layers(self.model)
        if layer_name == 'all':
            layers = all_layers
        elif layer_name in all_layers:
            layers = [layer_name]
        elif layer_name == 'middle':
            l = len(all_layers)
            layers = all_layers[((l//2)-1):((l//2)+7)]
        else:
            print('Error: Invalid layer name')
            return
        # 4. Image prediction probabilities and predicted_class
        predictions = self.model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions)
        # 5 Label to visualize
        nb_classes = self.get_model_nb_classes(
            self.model)  # 5.1 Model's number of classes
        if label == -1:
            class_to_visualize = predicted_class
        elif label < nb_classes and label > -1:
            class_to_visualize = label
        else:
            print('Error: Invalid label value')
            return
        # 6. Choose Visualization method
        all_methods = ['CAM_IMAGE_JET', 'CAM_IMAGE_BONE',
                       'CAM_AS_WEIGHTS', 'JUST_CAM_JET', 'JUST_CAM_BONE']
        if method_name in all_methods:
            method = Method[method_name]
        elif guided:
            method_name = 'GUIDED'
        else:
            print('Error: Invalid visualization method')
            return
        # TODO: Handler with dataset folder
        cams = []
        # 8. Iterate over layers to visualize
        for layer_to_visualize in layers:
            # 8.1 Get cam
            model = self.model
            cam = grad_cam(model, preprocessed_image,
                           class_to_visualize, layer_to_visualize, nb_classes)

            # 8.2 Generate visualization
            if guided:
                cam = guided_grad_cam(
                    self.model, cam, layer_to_visualize, preprocessed_image)
                cam_heatmap = create_guided_cam_image(cam, image, cam_rate=1)
            else:
                cam_heatmap = create_cam_image(cam, image, method)

            cams.append(Cam(image=cam_heatmap, target=class_to_visualize,
                            layer=layer_to_visualize, method=method_name, file_name=image_file))

        cams.insert(0, Cam(image=convert_to_bgr(image), target=class_to_visualize,
                           layer='Original', method=method_name, file_name=image_file))

        # Vai plotar a imagem da visualização de cada layer.
        plot(cams, image_file)
        # Faz o caminho simplificado de onde a imagem será salva, que o servidor usa para plotar no front.
        fname = extract_file_name(image_file)+'.png'
        path_view = os.path.join(os.path.dirname(
            __file__), '..', 'images', str(fname))
        # Salva a imagem no servidor, com o caminho completo.
        plt.savefig(path_view)
        plt.clf()  # Desaloca o espaço da imagem da memória.
        # print(path_view)  # Debug.
        # Retorna o caminho onde o arquivo foi salvo para que o servidor coloque na página.
        return path_view
