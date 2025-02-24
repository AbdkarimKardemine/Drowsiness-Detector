import cv2
import torch
from ultralytics import YOLO

# Charger le modèle YOLOv8
model = YOLO("C:/Users/karim krd/Downloads/best.pt")

# Ouvrir la webcam (0 = webcam par défaut)
cap = cv2.VideoCapture(0)

# Définir la résolution de la webcam (facultatif, dépend du matériel)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()  # Lire une image de la webcam
    
    if not ret:
        break  # Si aucun frame n'est capturé, quitter la boucle

    # Exécuter la détection YOLOv8
    results = model(frame, conf=0.5)  # Détection avec une confiance de 50%

    # Dessiner les résultats sur l'image
    annotated_frame = results[0].plot()

    # Afficher la sortie en temps réel
    cv2.imshow("Webcam YOLOv8", annotated_frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libérer la webcam et fermer les fenêtres OpenCV
cap.release()
cv2.destroyAllWindows()
