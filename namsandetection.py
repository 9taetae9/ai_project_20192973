import sys
import torch
import os
import cv2

def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        sys.exit(1)
    return torch.load(model_path)

def process_image(model, image_path, confidence_threshold):
    prediction = model.predict(image_path, confidence=0, overlap=30).json()
    predictions = prediction["predictions"]
    highest_confidence_pred = max(predictions, key=lambda x: x["confidence"]) if predictions else None

    if highest_confidence_pred:
        print(f"Highest confidence prediction: {highest_confidence_pred['confidence']}")

        if highest_confidence_pred["confidence"] * 100 > confidence_threshold:
            image = cv2.imread(image_path)
            x, y, width, height = highest_confidence_pred["x"], highest_confidence_pred["y"], highest_confidence_pred["width"], highest_confidence_pred["height"]
            topleft = (int(x - width / 2), int(y - height / 2))
            bottomright = (int(x + width / 2), int(y + height / 2))
            cv2.rectangle(image, topleft, bottomright, (0, 255, 0), 5)
            return image
        else:
            print("No prediction above the confidence threshold.")
            return None
    else:
        print("No predictions found.")
        return None

def display_image(image):
    if image is not None:
        cv2.namedWindow("Namsan Tower Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Namsan Tower Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

model_path = "namsan.pt"
model = load_model(model_path)

image_path = input("Enter the image file path: ")
if not os.path.exists(image_path):
    print("Image file not found")
    sys.exit(1)

confidence_threshold = 40
image_with_prediction = process_image(model, image_path, confidence_threshold)
display_image(image_with_prediction)
