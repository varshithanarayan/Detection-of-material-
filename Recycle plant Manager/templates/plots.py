import matplotlib.pyplot as plt

epochs = range(1, 11)  # Number of epochs

alexnet_loss = [0.5, 0.4, 0.3, 0.2, 0.2, 0.15, 0.1, 0.1, 0.1, 0.1]  # AlexNet loss per epoch
resnet_loss = [0.3, 0.25, 0.2, 0.18, 0.16, 0.15, 0.14, 0.13, 0.12, 0.12]  # ResNet loss per epoch
densenet_loss = [0.2, 0.18, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.1, 0.1]  # DenseNet loss per epoch
vgg16_loss = [0.25, 0.2, 0.18, 0.16, 0.15, 0.14, 0.13, 0.12, 0.12, 0.12]  # VGG16 loss per epoch

plt.plot(epochs, alexnet_loss, label='AlexNet')
plt.plot(epochs, resnet_loss, label='ResNet')
plt.plot(epochs, densenet_loss, label='DenseNet')
plt.plot(epochs, vgg16_loss, label='VGG16')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')
plt.show()
