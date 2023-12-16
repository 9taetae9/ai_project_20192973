import torch
import sys
import os
import cv2

model_path = "namsan.pt"
if os.path.exists(model_path):
    model = torch.load(model_path)

if len(sys.argv) < 2:
    print("Usage: python namsandetection.py <image file>")
    sys.exit(1)
    
image_path = sys.argv[1]

if not os.path.exists(image_path):
    print("Image file not found")
    sys.exit(1)
    
prediction = model.predict(image_path, confidence=0, overlap=30).json()

predictions = prediction["predictions"]

highest_confindence_pred = max(predictions, key=lambda x: x["confidence"]) if predictions else None

if highest_confindence_pred:
    print(f"Highest confidence prediction: {highest_confindence_pred['confidence']}")
    
image = cv2.imread(image_path)

confidence_trheshold = 40
if highest_confindence_pred and highest_confindence_pred["confidence"]*100 > confidence_trheshold:
    x = highest_confindence_pred["x"]
    y = highest_confindence_pred["y"]
    width = highest_confindence_pred["width"]
    height = highest_confindence_pred["height"]
    
    topleft = (int(x-width/2), int(y-height/2))
    bottomright = (int(x+width/2), int(y+height/2))
    
    cv2.rectangle(image, topleft, bottomright, (0, 255, 0), 5)
    
cv2.namedWindow("Namsan Tower Detection", cv2.WINDOW_NORMAL)
cv2.imshow("Namsan Tower Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()