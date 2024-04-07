import cv2, os
from ultralytics import YOLO

# Open the video file
root_folder = "MOT17/train"
directories = [d for d in os.listdir(root_folder)]

for d in directories:
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    frames_path = f"{root_folder}/{d}/img1"
    output_file = f"data/trackers/mot_challenge/MOT17-train/yolo/data/{d}.txt"

    image_files = [f for f in os.listdir(frames_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Sort the files by their digits, take off the word "frame" and the extension
    image_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))


    if os.path.exists(output_file):
        os.remove(output_file)

    for i, frame in enumerate(image_files):

        frame = cv2.imread(os.path.join(frames_path, image_files[i]))

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, classes=[0], conf=0.3)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Tensor to numpy array
        boxes = results[0].boxes


        # Save the bounding boxes and the frame in a data/labels.txt file
        with open(output_file, "a") as file:
            for j in range(len(boxes.xyxy)):
                if boxes.id != None:
                    file.write(f"{i + 1},{boxes.id[j]:.5f},{boxes.xyxy[j][0]:.5f},{boxes.xyxy[j][1]:.5f},{boxes.xywh[j][2]:.5f},{boxes.xywh[j][3]:.5f},1,-1,-1,-1\n")

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
