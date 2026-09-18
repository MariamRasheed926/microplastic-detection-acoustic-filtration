from ultralytics import YOLO
import cv2

url = "https://192.168.1.2:8080/video"

model = YOLO("microplastic_detection1.pt")

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 750)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 750)

skip_frames = 5  
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    frame_counter += 1
    if frame_counter % (skip_frames + 1) != 0:
        continue

    results = model.predict(source=frame, save=False, imgsz=750, conf=0.25)

    annotated_frame = results[0].plot()

    cv2.imshow("Microplastic Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
