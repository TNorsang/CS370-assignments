import torch
import numpy as np
import cv2
from pytube import YouTube
import os
from ultralytics import YOLO  # Make sure this import matches the actual library

# Load the YOLOv8 pre-trained model
model = YOLO("yolov8n.pt")

def download_youtube_video(youtube_link):
    yt = YouTube(youtube_link)
    video_stream = yt.streams.get_highest_resolution()
    video_stream.download(output_path="Videos")

def detect_and_save_objects(frame, detected_frame_path, model):
    # Perform object detection
    results = model(frame)
    
    # Assuming results contains the detections directly
    if hasattr(results, 'pred'):  # Checking if the 'pred' attribute exists
        detections = results.pred[0]  # Accessing the predictions
    else:
        print("The results object does not have the expected attributes.")
        return

    # Iterate through detections
    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = map(int, xyxy)  # Convert coordinates to integers
        conf = round(float(conf), 2)  # Round confidence to 2 decimal places
        cls = int(cls)  # Convert class index to integer
        
        # Access class name using class index
        cls_name = results.names[cls]
        
        # Format label with class name and confidence
        label = f'{cls_name} {conf:.2f}'
        
        # Draw bounding box and label on the frame
        plot_one_box([x1, y1, x2, y2], frame, label=label, color=(255, 0, 0), line_thickness=3)
    
    # Save the frame with drawn detections
    cv2.imwrite(detected_frame_path, frame)



def plot_one_box(xyxy, img, color=(128, 128, 128), label=None, line_thickness=3):
    xyxy = [int(x) for x in xyxy]
    cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thickness=line_thickness)
    if label:
        cv2.putText(img, label, (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def split_video_into_frames(video, model):
    video_title = os.path.splitext(os.path.basename(video))[0].replace('/', '-').replace('|', '-')  # Sanitize title
    frames_output_folder = os.path.join("Frames", video_title)
    objects_detected_folder = os.path.join("Objects_Detected", video_title)
    os.makedirs(frames_output_folder, exist_ok=True)
    os.makedirs(objects_detected_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    every_nth_frame = max(1, total_frames // 100)

    frame_count = 0
    saved_frames = 0

    while cap.isOpened() and saved_frames < 100:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % every_nth_frame == 0:
            original_frame_path = os.path.join(frames_output_folder, f"frame_{saved_frames}.jpg")
            cv2.imwrite(original_frame_path, frame)
            
            detected_frame_path = os.path.join(objects_detected_folder, f"detected_frame_{saved_frames}.jpg")
            detect_and_save_objects(frame, detected_frame_path, model)
            
            saved_frames += 1

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

def main():
    youtube_links = [
        "https://www.youtube.com/watch?v=WeF4wpw7w9k",
        "https://www.youtube.com/watch?v=2NFwY15tRtA&t=9s",
        "https://www.youtube.com/watch?v=5dRramZVu2Q&t=48s"
    ]
    for link in youtube_links:
        download_youtube_video(link)
        video_title = YouTube(link).title.replace('/', '-').replace('|', '-')  # Sanitize title
        video_file = os.path.join("Videos", f"{video_title}.mp4")
        split_video_into_frames(video_file, model)

if __name__ == "__main__":
    main()
