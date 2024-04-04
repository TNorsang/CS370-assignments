import torch
import numpy as np
import cv2
from pytube import YouTube
import os
from ultralytics import YOLO  # Make sure this import matches the actual library

# Load the YOLOv8 pre-trained model
model = YOLO("Norsang_Model.pt")

def norsang_model():
    


def download_youtube_video(youtube_link):
    yt = YouTube(youtube_link)
    video_stream = yt.streams.get_highest_resolution()
    video_stream.download(output_path="Videos")



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
