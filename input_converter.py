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
    
    
    return np.array(data["weight"]), np.array(data["bias"])

def save_to_json(weight, bias):
    
    with open("model.json", "w") as file:
        data = {
            "weight" : weight.tolist(),
            "bias" : bias.tolist()
        }    
        json.dump(data, file)


def multiply_weight(inputs, weight):
    return weight @ inputs

def add_bias(weighted_inputs, bias):  
    return np.add(weighted_inputs, bias)

def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)
    z_exp = np.exp(z)
    return z_exp / np.sum(z_exp, axis=0, keepdims=True)

def get_image():
    while True:
        with open("train_labels.csv", newline='') as file:
            data = list(csv.DictReader(file))
        
        while True:
            random.shuffle(data)
            for row in data:
                yield row
            
def create_batch(size=64):
    
    images = get_image()

    while True:
        labels = []
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
            labels.append(label)
        batch_array = np.transpose(np.array(batch))
        label_array = np.array(labels)
        yield batch_array, label_array

def get_Y_true(labels):
    
    Y_true_array = []
    for label in labels:
        Y_true = [0] * 10
        Y_true[int(label)] = 1
        Y_true_array.append(Y_true)
    return np.transpose(np.array(Y_true_array))

def cross_entropy_loss(Y_true, prediction):
    eps = 1e-12
    prediction = np.clip(prediction, eps, 1 - eps)

    N = prediction.shape[1]
    loss = -np.sum(Y_true * np.log(prediction)) / N
    return loss

def compute_gradients(Y_true, Y_pred, input):
    loss_gradient = get_loss_gradient(Y_true, Y_pred)
    weight_gradient = get_weight_gradient(loss_gradient, input)
    bias_gradient = get_bias_gradient(loss_gradient)

    return weight_gradient, bias_gradient

def get_loss_gradient(Y_true, prediction):
    return prediction - Y_true

def get_weight_gradient(loss_gradient, input):
    N = input.shape[1]

    return np.dot(loss_gradient, input.T) / N

def get_bias_gradient(loss_gradient):
    N = loss_gradient.shape[1]
    return np.sum(loss_gradient, axis=1, keepdims=True) / N
 
def apply_gradient(weight_gradient, bias_gradient, weight, bias):
    learning_rate = 0.01

    weight = weight - weight_gradient * learning_rate
    bias = bias - bias_gradient * learning_rate
    return weight, bias

def display_answer(predictions, labels):
    predictions = np.transpose(predictions)
    answers = np.argmax(predictions, axis=1)
    correct = 0
    for i, pred in enumerate(answers):
        print(f"{labels[i]} : {pred}")
        if int(labels[i]) == int(pred):
            correct += 1
    print(correct)

def run():
    weight, bias = load_from_json()
    bias = bias.reshape((10,1))
    batch_generator = create_batch()
    run = False
    while not run :
        inputs, labels = next(batch_generator)
        Y_true = get_Y_true(labels)
        N = 64
        weighted_inputs = multiply_weight(inputs, weight)
        activation_inputs = add_bias(weighted_inputs, bias)
        predictions = softmax(activation_inputs)

        
        display_answer(predictions, labels)
        print(cross_entropy_loss(Y_true, predictions))
        run = input()
        
        weight_gradient, bias_gradient = compute_gradients(Y_true, predictions, inputs)
        weight, bias = apply_gradient(weight_gradient, bias_gradient, weight, bias)
    save_to_json(weight, bias)

    
    


run()
