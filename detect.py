from ultralytics import YOLO
import cv2
import math

model = YOLO("best_ssaqta.pt")

video = cv2.VideoCapture("video2.mp4")

classNames = ["flame"]

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video2.mp4', fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = video.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            confidence = math.ceil((box.conf[0] * 100)) / 100
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            cls = int(box.cls[0])

            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (0, 0, 255)
            thickness = 2

            cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)

            print("Confidence --->", confidence)
            print("Class name -->", classNames[cls])

    out.write(frame)

video.release()
out.release()
cv2.destroyAllWindows()
