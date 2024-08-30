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
def save_model():
    global model
    alexnet_model= '../Models/Alexnet_model.h5'
    densenet_model= '../Models/Densenet_model.h5'
    resnet_model = '../Models/Resnet50_model.h5'
    vgg16_model = '../Models/VGG16_model.h5'
    if os.path.exists(alexnet_model) and os.path.exists(densenet_model) and os.path.exists(resnet_model) and os.path.exists(vgg16_model):
        return True
    else:
        return False
def train():
    import cv2 
    import os
    import gc
    import numpy as np
    from sklearn.model_selection import train_test_split
    from tensorflow import keras
    import pandas as pd
    from IPython.display import display
    import matplotlib.pyplot as plt

    shape = 224
    classes = []
    class Dataset:
        def __init__(self, shape):
            self.path = "../Dataset/data"
            self.data_files = os.listdir(self.path)
            self.data_csv = 0
            self.x_data = []
            self.y_bbox_data = []
            self.y_label_data = []
            self.x_data_files = []
            self.y_data_files = []
            self.labels = []
            self.shape = shape
            
        def __preprocessing_image(self, path):
            image = cv2.imread(path)
            image = cv2.resize(image, (shape, shape))
            image = np.reshape(image, (shape, shape, 3))
            image = np.array(image, dtype = "float32")
            image /= 255.0
            
            return image
            
        def __seperating_images_labels(self):
            for data_file in self.data_files:
                if data_file.split(".")[-1] == "txt":
                    data = open(f"{self.path}/{data_file}", "r").readline().split()
                    df = pd.DataFrame()
                    df["filename"] = [data_file]
                    df["class"] =[float(data[0])]
                    df["xmin"] = [float(data[1])]
                    df["ymin"] = [float(data[2])]
                    df["xmax"] = [float(data[3])]
                    df["ymax"] = [float(data[4])]
                    self.y_data_files.append(df)
        
            self.data_csv = pd.concat(self.y_data_files)
            for i in list(self.data_csv.columns[2:]):
                self.data_csv[i] = self.data_csv[i].astype(float)
                
        def __extract_images(self):
            self.images_paths = self.data_csv["filename"].values
            for image in self.images_paths:
                self.x_data.append(self.__preprocessing_image(self.path + "/" + image.split(".")[0] + ".jpeg"))
                
        def __extract_labels(self):
            global classes
            self.classes = self.data_csv["class"].unique().tolist()
            classes = self.classes
            self.y_label_data = self.data_csv["class"].values
            self.y_label_data = [self.classes.index(i) for i in self.y_label_data]
            self.y_label_data = keras.utils.to_categorical(self.y_label_data)
            self.y_label_data = np.array(self.y_label_data, dtype = "float32")
            
        def __extract_bboxes(self):
            for index, _ in enumerate(self.data_csv["xmin"].values):
                self.y_bbox_data.append(self.data_csv.iloc[[index]].values[0])
            self.y_bbox_data = np.array(self.y_bbox_data, dtype= "float32")
                
        def load_data(self):
            self.__seperating_images_labels()
            self.__extract_labels()
            self.__extract_images()
            self.data_csv = self.data_csv.drop('class', axis = 1)
            self.data_csv = self.data_csv.drop('filename', axis = 1)
            self.__extract_bboxes()
            
            return train_test_split(self.x_data, self.y_label_data, self.y_bbox_data, random_state = 42, shuffle = True)

    dataset = Dataset(shape)
    split = dataset.load_data()
    (x_train, x_test) = split[:2]
    (y_class_train, y_class_test) = split[2:4]
    (y_bbox_train, y_bbox_test) = split[4:]
    x_train = np.array(x_train, dtype = "float32")
    x_test = np.array(x_test, dtype = "float32")
    #y_bbox_train = np.reshape(y_bbox_train, (y_bbox_train.shape[0], 4))
    #y_bbox_test =  np.reshape(y_bbox_test,  (y_bbox_test.shape[0], 4))
    y_bbox_train = np.array(y_bbox_train, dtype = "float32")
    y_bbox_test = np.array(y_bbox_test, dtype = "float32")
    y_class_train = np.array(y_class_train, dtype = "float32")
    y_class_test = np.array(y_class_test, dtype = "float32")

    print(f"X_Train Shape : {x_train.shape}")
    print(f"X_Test Shape : {x_test.shape}")
    print(f"Y_class_Train Shape : {y_class_train.shape}")
    print(f"Y_class_Test Shape : {y_class_test.shape}")
    print(f"Y_bbox_Train Shape : {y_bbox_train.shape}")
    print(f"Y_bbox_Test Shape : {y_bbox_test.shape}")

    del dataset
    del split
    gc.collect()
    class Model():
        def __init__(self, cnn):
            self.model_input1 = keras.layers.Input((shape, shape, 3))
            self.cnn = cnn(weights = "imagenet", include_top = False, input_tensor = self.model_input1)
            for layer in self.cnn.layers:
                layer.trainable = False
            self.__build_model()
            self.plot_loss_acc()
            
        def __build_model(self):
            
            flatten = keras.layers.Flatten()(self.cnn.output)
            x = keras.layers.Dense(1024, activation = "relu", kernel_initializer ="he_normal")(flatten)
            x = keras.layers.Dense(512, activation = "relu", kernel_initializer = "he_normal")(x)
            x = keras.layers.Dense(256, activation = "relu", kernel_initializer = "he_normal")(x)
            x = keras.layers.Dense(128, activation = "relu", kernel_initializer = "he_normal")(x)
            output_1 = keras.layers.Dense(len(classes), activation = "softmax", name = "Class")(x)
            
            x = keras.layers.Dense(256, activation = "relu", kernel_initializer ="he_normal")(flatten)
            x = keras.layers.Dense(128, activation = "relu", kernel_initializer ="he_normal")(x)
            x = keras.layers.Dense(64, activation = "relu", kernel_initializer = "he_normal")(x)
            x = keras.layers.Dense(32, activation = "relu", kernel_initializer = "he_normal")(x)
            output_2 = keras.layers.Dense(4, activation = "sigmoid", name = "BBOX")(x)
            
            optimizer = keras.optimizers.Adam(1e-4)
            self.model = keras.models.Model(inputs = [self.model_input1], outputs = [output_1, output_2])
            losses = {"Class": "categorical_crossentropy","BBOX": "mse"}
            lossWeights = {"Class": 1.0, "BBOX": 1.0}
            self.model.compile(optimizer = optimizer, loss = losses, metrics = ["accuracy"])
            
            #display(keras.utils.plot_model(self.model))
            checkpoint = keras.callbacks.ModelCheckpoint("../model/weeddetectionmodel.h5")
            
            self.model.fit(x_train, y = {"Class":y_class_train, "BBOX":y_bbox_train},
                        epochs = 100, validation_data = (x_test, {"Class":y_class_test, "BBOX":y_bbox_test}),
                        shuffle = True, callbacks = [checkpoint])
        
        def plot_loss_acc(self):
            #Loss vs Epochs
            plt.plot(self.model.history.history['Class_loss'], label='train')
            plt.plot(self.model.history.history['val_Class_loss'], label='test')
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title(f"Loss vs Epochs")
            plt.legend()
            plt.show()
            #Accuracy vs Epochs
            plt.plot(self.model.history.history['Class_accuracy'], label='train')
            plt.plot(self.model.history.history['val_Class_accuracy'], label='test')
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.title(f"ACCURACY vs Epochs")
            plt.legend()
            plt.show()

    model = Model(keras.applications.resnet.ResNet101)
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Define AlexNet model
    class AlexNet(nn.Module):
        def __init__(self, num_classes=10):
            super(AlexNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    # Define VGG16 model
    class VGG16(nn.Module):
        def __init__(self, num_classes=10):
            super(VGG16, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    # Define DenseNet model
    class DenseNet(nn.Module):
        def __init__(self, num_classes=10):
            super(DenseNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Linear(512 * 7 * 7, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    # Initialize models
    alexnet = AlexNet().to(device)
    vgg16 = VGG16().to(device)
    densenet = DenseNet().to(device)

    # Set hyperparameters
    learning_rate = 0.001
    num_epochs = 10

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    alexnet_optimizer = optim.SGD(alexnet.parameters(), lr=learning_rate, momentum=0.9)
    vgg16_optimizer = optim.SGD(vgg16.parameters(), lr=learning_rate, momentum=0.9)
    densenet_optimizer = optim.SGD(densenet.parameters(), lr=learning_rate, momentum=0.9)

    # Training loop
    for epoch in range(num_epochs):
        # AlexNet training
        alexnet.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            alexnet_optimizer.zero_grad()
            outputs = alexnet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            alexnet_optimizer.step()

        # VGG16 training
        vgg16.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            vgg16_optimizer.zero_grad()
            outputs = vgg16(images)
            loss = criterion(outputs, labels)
            loss.backward()
            vgg16_optimizer.step()

        # DenseNet training
        densenet.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            densenet_optimizer.zero_grad()
            outputs = densenet(images)
            loss = criterion(outputs, labels)
            loss.backward()
            densenet_optimizer.step()

        # Print training loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Evaluation
        alexnet.eval()
        vgg16.eval()
        densenet.eval()
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = alexnet(images)
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

            accuracy = total_correct / total_samples * 100
            print(f"AlexNet Accuracy: {accuracy:.2f}%")

            total_correct = 0
            total_samples = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = vgg16(images)
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

            accuracy = total_correct / total_samples * 100
            print(f"VGG16 Accuracy: {accuracy:.2f}%")

            total_correct = 0
            total_samples = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = densenet(images)
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

            accuracy = total_correct / total_samples * 100
            print(f"DenseNet Accuracy: {accuracy:.2f}%")


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
