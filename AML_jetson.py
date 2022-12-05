from sklearn.preprocessing import LabelEncoder
from matplotlib import image
import numpy as np
import tensorflow as tf
import os
from PIL import Image
import os.path
import io
import base64
from tensorflow.keras.models import load_model 
import json
import argparse
from tensorflow.keras.metrics import sparse_top_k_categorical_accuracy

def top_5_accuracy(y_true,y_pred):
    return sparse_top_k_categorical_accuracy(y_true, y_pred, k=5)

def load_species(text_file):
    with open(text_file) as txt_file:
        species_dict = {}
        for line in txt_file.readlines():
            # split the lines to image path and label
            split_line = line.split(';')
            class_value = split_line[0]
            species_name = split_line[1].strip()

            species_dict[class_value] = species_name
        return species_dict

def decode_labels(all_labels, target_label):
  label_encoder = LabelEncoder()
  label_encoder.fit(all_labels)
  return label_encoder.inverse_transform(target_label)

def load_labels(file_txt):
    with open(file_txt) as txt_file:
        lines_arr = [line.strip() for line in txt_file.readlines()]
        labels = [line.split(' ')[1] for line in lines_arr]
        return labels

def preprocess_image(image_file):
    image_data = image.imread(image_file)
    image_data = tf.cast(image_data, tf.float32)/255
    image_data = tf.image.resize(image_data, (128, 128), method = "bilinear")
    image_data = np.array(image_data)
    image_data = image_data.ravel()
    image_data = image_data.reshape(1, 128, 128, 3)

    return image_data

def load_real_value(file_path):
    result = {}
    with open(file_path) as txt_file:
        for line in txt_file.readlines():
            splitted_line = line.split(" ")
            real_class = splitted_line[1].strip()

            file_name = splitted_line[0].split("/")[1]
            result[file_name] = real_class
            
    return result
        
def model_predict(model, image_data, labels):
    y_pred_score = model.predict(image_data)
    y_pred = np.argmax(y_pred_score, axis=1)
    return decode_labels(labels, y_pred)[0]

def check_if_file_exists(path):
    return os.path.exists(path)

def get_real_class(file_path):
    file_name = file_path.split('.')[0]
    print(file_name)
    return file_name




def main():
    parser = argparse.ArgumentParser(
        prog = 'AML',
        description = 'Classify Plant Species',
        epilog = 'Text at the bottom of help'
    )
    base_path = os.path.dirname(os.path.abspath(__file__))

    species_txt_path = os.path.join(base_path, "list", "species_list.txt")
    train_txt_path = os.path.join(base_path, "list", "train.txt")
    model_path = os.path.join(base_path, "saved_models_v2.4.0")

    ground_truth_path = os.path.join(base_path, "list", "groundtruth.txt")

    result_species = load_species(species_txt_path)
    result_labels = load_labels(train_txt_path)

    parser.add_argument('filename')
    

    args = parser.parse_args()
    test_path = os.path.join(base_path, "test", args.filename)

    if check_if_file_exists(test_path):
        image_data = preprocess_image(test_path)

        result_real = load_real_value(ground_truth_path)  
        
        model = load_model(model_path, custom_objects={'top_5_accuracy': top_5_accuracy}, compile=False)

        result = model_predict(model, image_data, result_labels)
        print("Real: {} - {}".format(result_real[args.filename], result_species[result_real[args.filename]]))
        print("Predicted: {} - {}".format(result, result_species[result]))
    else:
        print("Your file does not exists!")


if __name__ == '__main__':
    main()