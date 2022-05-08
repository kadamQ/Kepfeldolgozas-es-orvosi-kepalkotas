import os
import cv2
from keras.models import load_model
import imutils
from keras.preprocessing.image import img_to_array
import numpy as np

images_dir = os.listdir("./test_images")
camera = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier("./cascades/haarcascade_frontalface_default.xml")
model = load_model("./models/model2022-05-04.h5")

def detect_face_and_smile(image):
    image = imutils.resize(image, width=1150)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_copy = image.copy()
    rectangles = face_detector.detectMultiScale(gray,
                                                scaleFactor=1.2,
                                                minNeighbors=6,
                                                minSize=(30, 30))
    for (fx, fy, f_weight, f_height) in rectangles:
        region_of_interest = gray[fy:fy + f_height, fx:fx + f_weight]
        region_of_interest = cv2.resize(region_of_interest, (28, 28))
        region_of_interest = img_to_array(region_of_interest.astype("float") / 255.0)
        region_of_interest = np.expand_dims(region_of_interest, axis=0)
        (not_smiling, smiling) = model.predict(region_of_interest)[0]

        if smiling > not_smiling:
            label = "Van mosoly " + str(smiling * 100)[:5] + "%"
            cv2.putText(image_copy, label, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            cv2.rectangle(image_copy, (fx, fy), (fx + f_weight, fy + f_height), (0, 255, 0), 2)
        else:
            label = "Nincs mosoly " + str(not_smiling * 100)[:5] + "%"
            cv2.putText(image_copy, label, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(image_copy, (fx, fy), (fx + f_weight, fy + f_height), (0, 0, 255), 2)

    return image_copy


if len(images_dir) == 0:
    while camera.isOpened():
        _, frame = camera.read()
        frame = imutils.resize(frame, width=800)
        frame = detect_face_and_smile(frame)
        cv2.imshow("Mosolyfelismeres valos idoben", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()

else:
    images_path = "./test_images/test_image.jpg"
    test_image = cv2.imread(images_path)
    test_image = detect_face_and_smile(test_image)
    cv2.imshow("Mosolyfelismeres kepen", test_image)
    cv2.imwrite("./result_images/test_image_with_detection.jpg", test_image)
    cv2.waitKey()
