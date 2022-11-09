'''
Start.py
Open webcam to input into facial emotions classifier
Author: https://github.com/arunponnusamy/cvlib/blob/master/examples/face_detection_webcam.py
'''
import cvlib as cv
import cv2
import torch


# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
            

# open webcam
webcam = cv2.VideoCapture(0)

emotions = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad'}

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

mps_device = torch.device('mps')
model = torch.load('../models/AlexNet_Scratch_dataAugment')
model.eval()
model = model.to(mps_device)

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()

    if not status:
        print("Could not read frame")
        exit()

    # apply face detection
    face, confidence = cv.detect_face(frame)

    # loop through detected faces
    for idx, f in enumerate(face):
        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # crop and resize image of face
        img = cv2.resize(frame[startY:endY,startX:endX], (224, 224), interpolation = cv2.INTER_AREA)

        # make prediction
        input = torch.tensor(img).view(-1, 3, 224, 224).float()
        input = input.to(mps_device)

        output = model(input)
        _, preds = torch.max(output, 1)
        print(preds)

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
        text = f'{emotions[preds.item()]}'

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write confidence percentage on top of face rectangle
        cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0), 3)

    # display output
    cv2.imshow("Real-time emotions detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources
webcam.release()
cv2.destroyAllWindows()        
