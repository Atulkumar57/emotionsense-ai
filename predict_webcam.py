import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from model import EMOTIONS, IMG_SIZE

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()

    model = load_model(args.model)
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            face = cv2.resize(roi, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
            face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)
            probs = model.predict(face, verbose=0)[0]
            idx = int(np.argmax(probs))
            label = EMOTIONS[idx] + " " + str(round(probs[idx] * 100, 1)) + "%"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("EmotionSense - press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
