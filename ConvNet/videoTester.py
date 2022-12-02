import os
import cv2
import numpy as np
from keras.models import model_from_json
import keras.utils as image
import matplotlib.pyplot as plt



#load model
model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('fer.h5')


face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)

fig = plt.figure()
emotion_values = ['angry', 'happy', 'neutral', 'sad']

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0), 2)
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # Only consider [angry, happy, sad, neutral] and find max indexed array
        # sub_predictions = [predictions[0][0], predictions[0][3], predictions[0][4], predictions[0][6]]
        # print(sub_predictions)
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'happy', 'sad', 'neutral')
        # emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        # generate bar chart of probabilities. Set the color of the prediction as blue
        plt.clf()
        colors = ['grey'] * 4
        colors[emotion_values.index(predicted_emotion)] = 'blue'
        barChartFig = plt.bar(emotion_values,predictions[0], color = colors, width=.4)
        plt.xlabel('Emotion')
        plt.ylabel('Probability')
        plt.ylim(0,1)

        fig.canvas.draw()

        # img is rgb, convert to opencv's default bgr
        barChartImg = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        cv2.imshow("Probabilites", barChartImg)

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,255,0), 3)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)


    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows