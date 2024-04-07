import cv2, os
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the images directory.
dataset_path = "MOT17/test"

for directory in os.listdir(dataset_path):
    images_path = os.path.join(dataset_path, directory, "img1")
    image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Sort the files by their digits, take off the word "frame" and the extension
    image_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    print(image_files)

    for i, frame in enumerate(image_files):
        frame = cv2.imread(os.path.join(images_path, image_files[i]))

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, classes=[0,41,42])

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Tensor to numpy array
        boxes = results[0].boxes.xyxy

        # Save the bounding boxes and the frame in a data/labels.txt file
        with open("labels.txt", "a") as file:
            for box in boxes:
                print(results[0].boxes.id)
                file.write(f"{i} {box[0]} {box[1]} {box[2]} {box[3]}\n")
                input()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        #else:
            # Break the loop if the end of the video is reached
            #break


# Release the video capture object and close the display window
cv2.destroyAllWindows()
