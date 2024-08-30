import cv2 
import os
import gc
import hashlib
import socket
import mysql.connector as mssql
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import os, sys
import random
import string
def getMachine_addr():
	os_type = sys.platform.lower()
	command = "wmic bios get serialnumber"
	return os.popen(command).read().replace("\n","").replace("	","").replace(" ","")

def getUUID_addr():
	os_type = sys.platform.lower()
	command = "wmic path win32_computersystemproduct get uuid"
	return os.popen(command).read().replace("\n","").replace("	","").replace(" ","")

def extract_command_result(key,string):
    substring = key
    index = string.find(substring)
    result = string[index + len(substring):]
    result = result.replace(" ","")
    result = result.replace("-","")
    return result
Dict = {
           0: {'Name' : 'Battery', 'Type' : 'Non Biodegradable', 'Disposal' : 'Bag them and throw in a bin in secure container'},
           1:{'Name' : 'Biological','Type' : 'Biodegradable', 'Disposal' : 'Put in any wet waste basket'},
           2: {'Name' : 'Brown Glass', 'Type' : 'Non Biodegradable', 'Disposal' : 'Collect the solvent as chemical waste, drain, let dry and dispose of glass in a glass waste box'},
           3: {'Name' : 'Cardboard', 'Type' : 'Biodegradable', 'Disposal' : 'Find a dry waste basket and dispose there safely'},
           4: {'Name' : 'Clothes', 'Type' : 'Biodegradable', 'Disposal' : 'Find a dry waste basket and dispose there safely'}, 
           5: {'Name' : 'Green Glass', 'Type' : 'Non Biodegradable',  'Disposal' : 'Collect the solvent as chemical waste, drain, let dry and dispose of glass in a glass waste box'},
           6: {'Name' : 'Metal', 'Type' : 'Non Biodegradable',  'Disposal' : 'Find a Shredder, If not wrap with secure cover and dump in dry waste'},
           7: {'Name' : 'Paper',  'Type' : 'Biodegradable',  'Disposal' : 'Find a dry waste basket and dispose there safely'},
           8: {'Name' : 'Plastic',  'Type' : 'Non Biodegradable',  'Disposal' : 'Find a dry waste basket and dispose there safely'},
           9: {'Name' : 'Shoes',  'Type' : 'Biodegradable',  'Disposal' : ' put them in a bring bank and put in a dry waste basket'},
           10: {'Name' : 'Trash',  'Type' :'Biodegradable', 'Disposal' : 'If medical waste, wrap in trashbag and put in wet waste, else in dry waste basket'},
           11: {'Name' : 'White Glass', 'Type' : 'Non Biodegradable', 'Disposal' : 'Collect the solvent as chemical waste, drain, let dry and dispose of glass in a glass waste box'},
           }
def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    
    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    print(preds)
    return Dict[int(preds)]

def get_ip_address_of_host():
    mySocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        mySocket.connect(('10.255.255.255', 1))
        myIPLAN = mySocket.getsockname()[0]
    except:
        myIPLAN = '127.0.0.1'
    finally:
        mySocket.close()
    return myIPLAN
def predict_alexnet(path):
    shape = 224
    classes = []
    model = load_model('../Models/Alexnet_model.h5')
    preds = model_predict(path, model)
    return preds

def predict_VGG16(path):
    shape = 224
    classes = []
    model = load_model('../Models/VGG16_model.h5')
    preds = model_predict(path, model)
    return preds

def predict_resnet(path):
    shape = 224
    classes = []
    model = load_model('../Models/Resnet50_model.h5')
    preds = model_predict(path, model)
    return preds

def predict_densenet(path):
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    import tensorflow_hub as hub
    model = tf.keras.models.load_model(
        ('../Models/Densenet_model.h5'),
        custom_objects={'KerasLayer':hub.KerasLayer}
    )
    image_path = path

    # Load the image and convert it to an array
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)

    # Preprocess the image (normalize and resize) to match the input requirements of MobileNetV2
    preprocessed_image = preprocess_input(image_array)

    # Add batch dimension to the preprocessed image
    input_image = np.expand_dims(preprocessed_image, axis=0)

    # Make the prediction
    prediction = model.predict([input_image, input_image])
    max_index = np.argmax(prediction)
    print(max_index) 
    print("Densenet")
    preds = Dict[int(max_index)]


    return preds

def md5(input_string):
    md5_hash = hashlib.md5()
    md5_hash.update(input_string.encode('utf-8'))
    return md5_hash.hexdigest()
def key_validate(str):
    conn = mssql.connect(
        user='root', password='root', host='localhost', database='garbage'
        )
    cur = conn.cursor()
    private_key = extract_command_result("SerialNumber",getMachine_addr()) + extract_command_result("UUID",getUUID_addr())
    if private_key in str:
        cur.execute("select * from SOFTKEY where private_key = %s and public_key = %s",(md5(private_key),md5(extract_command_result(private_key,str))))
        data=cur.fetchone()
        if data:
            return True
        else:
            return False
    else:
        return False
