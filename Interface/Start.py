'''
Start.py
Open webcam to input into facial emotions classifier
Author: https://github.com/arunponnusamy/cvlib/blob/master/examples/face_detection_webcam.py
'''
import cvlib as cv
import cv2
import torch

# open webcam
webcam = cv2.VideoCapture(0)
model = torch.load('../models/AlexNet_Scratch_dataAugment')
model.eval()
emotions = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad'}

if not webcam.isOpened():
    print("Could not open webcam")
    exit()
    

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()

    if not status:
        print("Could not read frame")
        exit()

    # apply face detection
    face, confidence = cv.detect_face(frame)

    # crop frame using face and convert to tensor
    outputs = model(face)
    _, preds = torch.max(outputs, 1)

    # loop through detected faces
    for idx, f in enumerate(face):
        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        text = "{:.2f}%".format(confidence[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write confidence percentage on top of face rectangle
        cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0), 2)

    # display output
    cv2.imshow("Real-time face detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources
webcam.release()
cv2.destroyAllWindows()        