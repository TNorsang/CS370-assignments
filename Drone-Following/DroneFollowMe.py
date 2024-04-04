import torch
import numpy as np
import cv2
from pytube import YouTube
import os
from ultralytics import YOLO  # Make sure this import matches the actual library

# Load the YOLOv8 pre-trained model
model = YOLO("Norsang_Model.pt")

def norsangModel():
    if os.path.exists("./runs/detect/predict") and len(os.listdir("./runs/detect/predict")) > 0:
        print("Prediction Video Already Exists! Check it out in runs/detect/predict folder!")
        return
    videos = ["./Videos/Cyclist and vehicle Tracking - 1.mp4", "./Videos/Cyclist and vehicle tracking - 2.mp4", "./Videos/Drone Tracking Video.mp4"]
    for video in videos:        
        model.predict(source=video, show=True, save=True, conf=0.5)
    print("Prediction Video Added! Check it out in runs/detect/predict folder!")


def download_youtube_video(youtube_link):
    if os.path.exists("Videos") and len(os.listdir("Videos")) > 0:
        print("Videos Already Downloaded! Lets Extract the Frames!")
        return
    yt = YouTube(youtube_link)
    video_stream = yt.streams.get_highest_resolution()
    video_stream.download(output_path="Videos")
    print("Video downloaded successfully! Lets Extract the Frames!")

def split_video_into_frames(video, model):

    if os.path.exists("Frames") and len(os.listdir("Frames")) > 0:
        print("Frames Already Exists! Let's Start Object Detection!")
        return
    video_title = os.path.splitext(os.path.basename(video))[0].replace('/', '-').replace('|', '-')  # Sanitize title
    frames_output_folder = os.path.join("Frames", video_title)
    os.makedirs(frames_output_folder, exist_ok=True)
    
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
            saved_frames += 1

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Frames Created! Let's Start Object Detection!")

def main():
    youtube_links = [
        "https://www.youtube.com/watch?v=WeF4wpw7w9k",
        "https://www.youtube.com/watch?v=2NFwY15tRtA&t=9s",
        "https://www.youtube.com/watch?v=5dRramZVu2Q&t=48s"
    ]
    for link in youtube_links:
        download_youtube_video(link)
        video_title = YouTube(link).title.replace('/', '-').replace('|', '-') 
        video_file = os.path.join("Videos", f"{video_title}.mp4")
        split_video_into_frames(video_file, model)
    
    norsangModel()

if __name__ == "__main__":
    main()