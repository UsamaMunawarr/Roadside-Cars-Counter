from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from sort import *
import os

#######################################
### Video Setup and Model Loading #####
#######################################
cap = cv2.VideoCapture("cars.mp4")
model = YOLO("yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread("mask-950x480.png")

#######################################
### Tracker Setup and Counting Line ###
#######################################
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [370, 297, 750, 297]  # Counting line coordinates
totalCount = []

#######################################
### VideoWriter Setup #################
#######################################
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

current_dir = os.getcwd()
output_path = os.path.join(current_dir, "output_cars.mp4")

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

#######################################
### Main Loop #########################
#######################################
while True:
    success, img = cap.read()
    if not success:
        break  # End of video

    # Apply mask
    mask_resized = cv2.resize(mask, (frame_width, frame_height))
    imgRegion = cv2.bitwise_and(img, mask_resized)

    # Overlay graphics
    imageGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    if imageGraphics is not None:
        imageGraphics = cv2.resize(imageGraphics, (400, 120))
        img = cvzone.overlayPNG(img, imageGraphics, pos=(0, 0))
        cv2.putText(img, str(len(totalCount)), (200, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # YOLO detection
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update tracker
    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    # Draw tracked boxes and count
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=1, offset=3)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Counting logic
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[3] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # Write frame to output video
    out.write(img)
    cv2.imshow("Car Counter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#######################################
### Release Everything ################
#######################################
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output video saved at: {output_path}")
