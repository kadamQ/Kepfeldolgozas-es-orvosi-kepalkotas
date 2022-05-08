import cv2
import imutils
from keras.preprocessing.image import img_to_array
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from lenet.nn.conv import LeNet
from sklearn.metrics import classification_report
import time
import matplotlib.pyplot as plt

data = []
labels = []
epochs = 14

for path_of_image in sorted(list(imutils.paths.list_images("./smiles"))):
    image = cv2.imread(path_of_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)
    label = path_of_image.split(os.path.sep)[-3]
    if label == "positives":
        label = "smiling"
    else:
        label = "not_smiling"
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)
all_class = labels.sum(axis=0)

class_weight = {
    0: all_class.max() / all_class[0],
    1: all_class.max() / all_class[1]
}

(x_train, x_test, y_train, y_test) = train_test_split(data,
                                                      labels,
                                                      test_size=0.20,
                                                      stratify=labels)

print("Modell létrehozása")
model = LeNet.build(width=28,
                    height=28,
                    depth=1,
                    classes=2)
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

print("Modell tanítása")
training = model.fit(x_train, y_train,
                     validation_data=(x_test, y_test),
                     class_weight=class_weight,
                     batch_size=64,
                     epochs=epochs,
                     verbose=1)

print("Modell kiértékelése")
predictions = model.predict(x_test, batch_size=64)
print(classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=le.classes_))

print("Modell mentése")
today = time.strftime("%Y-%m-%d")
model.save(os.getcwd() + "/models/model" + today + ".h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), training.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), training.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), training.history["accuracy"], label="acc")
plt.plot(np.arange(0, epochs), training.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('./plots/loss_and_accuracy.pdf')
plt.show()
