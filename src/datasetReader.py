import numpy as np
import tensorflow as tf
from tensorflow import keras
import os


class DatasetReader:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.shapeImageFeatures = []  # Features of the shape image
        self.shapeClassName = []  # Shape class name before convert to numerical encode
        self.shapeTargetName = []  # Shape class name after to numerical
        self.signImageFeatures = []  # Features of the sign image
        self.signClassName = []  # Sign class name before convert to numerical encode
        self.signTargetName = []  # Sign class name after convert to numerical

    def create_shape_dataset(self):
        for dir1 in os.listdir(self.image_folder):
            if not (dir1.startswith('.') or dir1.endswith('.txt')):
                for dir2 in os.listdir(
                        os.path.join(self.image_folder, dir1)):  # Each of the directory has a sub directory
                    if not (dir2.startswith('.') or dir2.endswith('.txt')):
                        for file in os.listdir(os.path.join(self.image_folder, dir1, dir2)):
                            if not file.startswith('.'):
                                image_path = os.path.join(self.image_folder, dir1, dir2, file)
                                image = keras.preprocessing.image.load_img(image_path)
                                image = keras.preprocessing.image.img_to_array(image)
                                image = image.astype('float32')
                                self.shapeImageFeatures.append(image)
                                self.shapeClassName.append(dir1)
                                self.signImageFeatures.append(image)
                                self.signClassName.append(dir2)

    def label_encoder(self):
        self.shapeTargetName = {k: v for v, k in enumerate(np.unique(self.shapeClassName))}
        self.signTargetName = {k: v for v, k in enumerate(np.unique(self.signClassName))}
