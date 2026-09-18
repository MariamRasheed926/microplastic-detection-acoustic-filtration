from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO("microplastic_detection1.pt")

# Open a connection to the webcam (use 0 for the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set the video frame width and height (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 750)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 750)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Perform inference on the frame
    results = model.predict(source=frame, save=False, imgsz=750, conf=0.25)  # Use save=False to avoid saving files

    # Annotate the frame with predictions
    annotated_frame = results[0].plot()  # `plot` automatically draws predictions on the frame

    # Display the annotated frame
    cv2.imshow("Microplastic Detection", annotated_frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
