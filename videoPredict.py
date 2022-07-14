from xml.etree.ElementTree import tostring
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
import cv2
import json
import numpy as np


def start_video():
    face_classifier = cv2.CascadeClassifier(
        'models/haarcascade_frontalface_default.xml')
    classifier = load_model('models/emotion_model.h5')
    model_gender = cv2.face.FisherFaceRecognizer_create()
    model_gender.read('models/gender_model.xml')

    genders = ["female", "male"]

    emotion_labels = ['Angry', 'Disgust', 'Fear',
                      'Happy', 'Neutral', 'Sad', 'Surprise']

    cap = cv2.VideoCapture(0)
    attemp = 0
    expressions_dictionary = {"emotions": {'Angry': 0, 'Disgust': 0, 'Fear': 0,
                                           'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Surprise': 0}, "gender": {"male": 0, "female": 0}}
    padding = 120

    while True:
        _, frame = cap.read()
        print("attemp: " + str(attemp))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)
        attemp += 1

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cropped_face = gray[y:y+h, x:x+w]
            gender_face = cv2.resize(cropped_face, (350, 350))
            resized_face = cv2.resize(
                cropped_face, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([resized_face]) != 0:
                roi = resized_face.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                gender_prediction = model_gender.predict(gender_face)
                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                gender = genders[gender_prediction[0]]
                expressions_dictionary["emotions"][label] += 1
                expressions_dictionary["gender"][gender] += 1
                label_position = (x, y - 20)
                emotion_position = (x + padding, y - 20)
                cv2.putText(frame, label, label_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, gender, emotion_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    json_object = json.dumps(expressions_dictionary, indent=4)
    with open("emotion.json", "w") as outfile:
        outfile.write(json_object)
    cap.release()
    cv2.destroyAllWindows()
