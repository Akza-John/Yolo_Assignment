from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Load the input video
video_path = './peoplecount.mp4'
cap = cv2.VideoCapture(video_path)

# Print error if the video is unable to open or not found
if not cap.isOpened():
    print("Error: Couldn't open video file or video is unavailable.")
    exit()

# read frames
while True:
    ret, frame = cap.read()

    if not ret:
        # End of video
        break

    # Detect and track objects
    results = model.track(frame, persist=True)

    # Plot results
    frame_with_boxes = results[0].plot()

    # Visualize
    cv2.imshow('Frame', frame_with_boxes)

    # Exit on 'q' key press
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
