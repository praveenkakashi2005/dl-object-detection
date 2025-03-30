import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F

# Load pre-trained Faster R-CNN model
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

true_positives = 0
false_positives = 0
false_negatives = 0
total_detections = 0

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
    
    for i, box in enumerate(boxes):
        if scores[i] > 0.6:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box)
            label = COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
            score = scores[i]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Text background
            text = f"{label} {score:.2f}"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), (0, 0, 0), -1)

            # Put label text
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            detected_objects.append(f"{label} ({score:.2f})")

            # Accuracy calculations
            true_positives += 1
        else:
            false_positives += 1

    total_detections += len(detected_objects)
    false_negatives = max(0, total_detections - true_positives)

    # Precision, Recall, Accuracy calculations
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    accuracy = (true_positives / total_detections) * 100 if total_detections > 0 else 0

    # Print detected objects and accuracy metrics
    if detected_objects:
        print("\n===== Detected Objects =====")
        for obj in detected_objects:
            print(obj)

        print(f"\nPrecision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Accuracy: {accuracy:.1f}%")

    # Save and display output
    cv2.imwrite("output.jpg", frame)
    print("Frame saved as output.jpg")

    # Show output in Matplotlib (avoiding OpenCV GUI issues)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
