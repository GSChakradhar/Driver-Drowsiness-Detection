import cv2
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os

cap = cv2.VideoCapture(0)

cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

detector = FaceDetector()

classifier = Classifier(
    r"C:\Users\chand\OneDrive\Documents\Desktop\AI&ML projects\Mini project 2\Model 2\keras_model.h5",
    r"C:\Users\chand\OneDrive\Documents\Desktop\AI&ML projects\Mini project 2\Model 2\labels.txt"
)


labels = ["Awake...", "Drowsy!!"]


offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    img_output = img.copy()
    img, faces = detector.findFaces(img)

    if faces:
        face = faces[0]
        x, y, w, h = face["bbox"]

        img_white = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        img_crop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if img_crop.size != 0:
            h_crop, w_crop = img_crop.shape[:2]
            aspect_ratio = h_crop / w_crop

            if aspect_ratio > 1:
                k = imgSize / h_crop
                w_cal = math.ceil(k * w_crop)
                img_resize = cv2.resize(img_crop, (w_cal, imgSize))
                w_gap = math.ceil((imgSize - w_cal) / 2)
                img_white[:, w_gap:w_gap + w_cal] = img_resize

            else:
                k = imgSize / w_crop
                h_cal = math.ceil(k * h_crop)
                img_resize = cv2.resize(img_crop, (imgSize, h_cal))
                h_gap = math.ceil((imgSize - h_cal) / 2)
                img_white[h_gap:h_gap + h_cal, :] = img_resize

            prediction, index = classifier.getPrediction(img_white, draw=False)
            label_text = labels[index]


            cv2.rectangle(img_output,(x - offset, y - offset - 70),(x - offset + 400, y - offset + 60 - 50),(0, 255, 0), cv2.FILLED)

            cv2.putText(img_output, label_text,
                        (x, y - 30),
                        cv2.FONT_HERSHEY_COMPLEX,
                        2, (0, 0, 0), 2)

    cv2.imshow("Image", img_output)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
