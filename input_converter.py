import numpy as np
from PIL import Image
import random
import json
import csv

def transform_image_to_vector(image):
    array = []

    with Image.open(image) as im:
        pixels = im.load()
        width, height = im.size
    
    for y in range(height):
        row = []
        for x in range(width):
            color = pixels[x ,y]
            simplified_color = color / 255
            row.append(simplified_color)
        array.append(row)
    
    array = np.array(array)
    vector = array.flatten()
    return vector

def load_from_json():
    with open("model.json", "r") as file:
        data = json.load(file)
    weight_matrix = []
    bias_vector = []
    for i in range(10):
        weight_matrix.append(data[f"weight{i}"])
        bias_vector.append(data[f"bias{i}"])
    return np.array(weight_matrix), np.array(bias_vector)

def multiply_weight(inputs, weight):
    return np.matmul(weight, inputs)

def add_bias(weighted_inputs, bias):  
    return np.add(weighted_inputs, bias)

def softmax(z):
    z = z - np.max(z)
    z_exp = np.exp(z)
    return z_exp / np.sum(z_exp)

def get_image():
    while True:
        with open("train_labels.csv", newline='') as file:
            data = list(csv.DictReader(file))
        
        while True:
            random.shuffle(data)
            for row in data:
                yield row
            

def create_batch(size=32):
    
    images = get_image()

    while True:
        label_array = []
        batch = []
        for i in range(size):
            image = next(images)
            filename, label = (
                image['filename'],
                image['label']
            )
            filename = rf"train\{filename}"
            image_vector = transform_image_to_vector(filename)
            batch.append(image_vector)
            label_array.append(label)
        yield np.array(batch), np.array(label_array)


def run():
    weight, bias = load_from_json()
    inputs = transform_image_to_vector(r"train\0.png")
    weighted_inputs = multiply_weight(inputs, weight)
    activation_inputs = add_bias(weighted_inputs, bias)
    outputs = softmax(activation_inputs)
    prediction = np.argmax(outputs)
    print(outputs)




