'''
Start.py
Open webcam to input into facial emotions classifier
Edited by: Gautam Mundewadi
Source: https://github.com/arunponnusamy/cvlib/blob/master/examples/face_detection_webcam.py
'''
import cvlib as cv
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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
fig = plt.figure()

emotions = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad'}
emotion_values = ['angry', 'happy', 'neutral', 'sad']

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

mps_device = torch.device('mps')
model = torch.load('../models/MobileNetV2_TL')
model.eval()
model = model.to(mps_device)

# loop through frames
while webcam.isOpened():

    width  = webcam.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

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

        l = max(endX - startX, endY - startY)
        leftAdjust = 40
        cropped_face = frame[startY:startY + l,startX - leftAdjust : startX + l-leftAdjust]
        
        # crop and resize image of face
        img = cv2.resize(cropped_face, (224, 224), interpolation=cv2.INTER_AREA)

        # make prediction
        input = torch.tensor(img).view(-1, 3, 224, 224).float()
        input = input.to(mps_device)

        output = model(input)
        softmax = torch.nn.Softmax(dim = 1)
        probabilites = softmax(output)

        # draw rectangle and prediction over face
        _, preds = torch.max(probabilites, 1)

        # generate bar chart of probabilities. Set the color of the prediction as blue
        plt.clf()
        colors = ['grey'] * 4
        colors[preds.item()] = 'blue'
        barChartFig = plt.bar(emotion_values,probabilites.cpu().detach().numpy()[0], color = colors, width=.4)
        plt.xlabel('Emotion')
        plt.ylabel('Probability')
        plt.ylim(0,1)

        # display matplotlib fig using opencv
        fig.canvas.draw()

        # img is rgb, convert to opencv's default bgr
        barChartImg = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        # cv2.resizeWindow(barChartImg, (int(width*.2), int(height*.2)))
        
    
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
        text = f'{emotions[preds.item()]}'

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write emotion on top of face rectangle
        cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0), 3)

    # display frames and bar chart
    cv2.imshow("Real-time emotions detection", frame)
    cv2.imshow("Probabilites", barChartImg)
    cv2.setWindowProperty("Real-time emotions detection", cv2.WND_PROP_FULLSCREEN, cv2.WND_PROP_FULLSCREEN)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources
webcam.release()
cv2.destroyAllWindows()        
