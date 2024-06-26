'''import cv2
import imutils
from imutils.video import VideoStream
rtsp_url = "rtsp://admin:Espe2024*@10.9.9.94:554/Streaming/Channels/101"
video_stream = VideoStream(rtsp_url).start()

while True:
    frame = video_stream.read()
    if frame is None:
        continue

    frame = imutils.resize(frame,width=1200)
    cv2.imshow('AsimCodeCam', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()'''

import cv2
import imutils
from imutils.video import VideoStream
import torch
import pathlib
import chime

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Cargar el modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path = 'model/best7997.pt')


# Configurar el modelo para inferencia
model.eval()

# Configurar la URL de la cámara RTSP
rtsp_url = "rtsp://admin:Espe2024*@192.168.100.81:554/Streaming/Channels/101"
video_stream = VideoStream(rtsp_url).start()

while True:
    # Leer el frame desde la transmisión de video
    frame = video_stream.read()
    if frame is None:
        continue

    # Redimensionar el frame
    frame = imutils.resize(frame, width=1280)

    # Convertir la imagen de BGR a RGB (YOLOv5 utiliza imágenes en formato RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Realizar la inferencia con YOLOv5
    results = model(rgb_frame)
    
    info = results.pandas().xyxy[0]
    print(info)
    # Dibujar las cajas delimitadoras en el frame original
    for pred in results.xyxy[0]:
        conf = float(pred[4])
        if conf > 0.5:  # Filtro de confianza
            xmin, ymin, xmax, ymax = map(int, pred[:4])
            label = f"{model.names[int(pred[5])]}: {conf:.2f}"  # Nombre y confianza
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if conf>= 0.95:  # Verificar si la confianza es mayor o igual a 0.93
                chime.theme('pokemon')
                chime.error(sync=True)

    # Mostrar el frame con las cajas delimitadoras
    cv2.imshow('Vision Safe v1.1', frame)

    # Comprobar si se presiona la tecla 'q' para salir del bucle
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Liberar recursos y cerrar ventanas
cv2.destroyAllWindows()
video_stream.stop()
