import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F

# Load pre-trained Faster R-CNN model with correct weights
weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

# COCO class labels
COCO_INSTANCE_CATEGORY_NAMES = weights.meta["categories"]

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to RGB and tensor
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        predictions = model(image_tensor)

    # Get bounding boxes, labels, and scores
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    detected_objects = []

    # Draw only high-confidence detections
    for i, box in enumerate(boxes):
        if scores[i] > 0.6:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box)
            label = COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
            score = scores[i]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Text background rectangle
            text = f"{label} {score:.2f}"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), (0, 0, 0), -1)

            # Put label text
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            detected_objects.append(f"{label} ({score:.2f})")

    # Print detected objects in the terminal
    if detected_objects:
        print("\n===== Detected Objects =====")
        for obj in detected_objects:
            print(obj)

    # Save frame with detections
    cv2.imwrite("output.jpg", frame)
    print("Frame saved as output.jpg")

    # Display the image using Matplotlib (Fixes OpenCV GUI issue)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
