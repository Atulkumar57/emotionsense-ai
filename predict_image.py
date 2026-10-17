import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from model import EMOTIONS, IMG_SIZE

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def predict(model, gray_face):
    face = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    probs = model.predict(face, verbose=0)[0]
    idx = int(np.argmax(probs))
    return EMOTIONS[idx], float(probs[idx]), probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    model = load_model(args.model)
    img = cv2.imread(args.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("No face detected.")
        return

    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        label, conf, _ = predict(model, roi)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label + " " + str(round(conf * 100, 1)) + "%", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out = "prediction.jpg"
    cv2.imwrite(out, img)
    print("Saved annotated image to", out)


if __name__ == "__main__":
    main()
