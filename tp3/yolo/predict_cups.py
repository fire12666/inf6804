import cv2, os
import supervision as sv
from ultralytics import YOLO


model = YOLO('YOLOv9e.pt')
tracker = sv.ByteTrack()
annotator = sv.BoxAnnotator()

frames_path = "yolo/frames"
output_file = f"labels.txt"
image_files = [f for f in os.listdir(frames_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Sort the files by their digits, take off the word "frame" and the extension
image_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

if os.path.exists(output_file):
    os.remove(output_file)

for i, frame in enumerate(image_files):

    frame = cv2.imread(os.path.join(frames_path, image_files[i]))

    # Run YOLOv8.
    results = model(frame, classes=[41], conf=0.1, iou=0.5)

    detections = sv.Detections.from_ultralytics(results[0])
    detections = tracker.update_with_detections(detections)

    labels = []

    with open(output_file, "a") as file:
        for bbox, _, confidence, class_id, tracker_id, _ in detections:
            if tracker_id is not None:
                labels.append(f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}")
                file.write(f"{i} {int(tracker_id)} {int(bbox[0])} {int(bbox[1])} {int(bbox[2] - bbox[0])} {int(bbox[3] - bbox[1])}\n")

    annotated_frame = annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)

    # Display the annotated frame
    cv2.imshow("YOLOv9 Tracking", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    #else:
        # Break the loop if the end of the video is reached
        #break

# Release the video capture object and close the display window
cv2.destroyAllWindows()
