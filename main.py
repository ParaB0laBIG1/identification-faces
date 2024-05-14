import cv2
import numpy as np

cap = cv2.VideoCapture(0)
faces = cv2.CascadeClassifier('data/faces.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = faces.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=4)
    
    for (x, y, w, h) in results:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
        cv2.putText(frame, 'Human', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("result", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
