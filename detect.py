# Import Librarys
import torch
import cv2
import numpy as np
import pandas
import pathlib
import chime


temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


# Read model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path = 'model/best.pt')

# Iniciamos VideoCapture
cap = cv2.VideoCapture("rtsp://admin:Espe2024*@192.168.100.81:554/Streaming/Channels/101")

# Loop
while True:
    # Leemos frame a frame
    ret, frame = cap.read()

    # Detectamos objetos
    results = model(frame)

    info = results.pandas().xyxy[0] #im1 predictions
    print(info)

    # Mostramos resultados
    cv2.imshow('Detector de Armas', np.squeeze(results.render()))

    if info['confidence'].max() >= 0.90:
        chime.theme('pokemon')
        chime.error(sync=True)

    # Si se presiona la tecla ESC se cierra el programa
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
