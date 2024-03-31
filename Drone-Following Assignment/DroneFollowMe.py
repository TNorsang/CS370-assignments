import torch
import numpy as np
import cv2
from pytube import YouTube
import os

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Downloading the youtube video
def download_youtube_video(youtube_link):
    yt = YouTube(youtube_link)
    video_stream = yt.streams.first()
    # Specifying the name of the folder where the Videos are stored in
    # Creating the Videos Folder

    # videos_output_path = os.path.join("Videos", f"{yt.title}.mp4")
    # os.makedirs(video_output_folder, exist_ok=True)
    video_stream.download(output_path="Videos")


# Splitting each youtube video into frames 
def split_video_into_frames(video):
    # Get video title without extension
    video_title = os.path.splitext(os.path.basename(video))[0]
    # Creating a folder for frames based on video title
    frames_output_folder = os.path.join("Frames", video_title)
    os.makedirs(frames_output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(frames_output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()


def main():
    youtube_links = ["https://www.youtube.com/watch?v=WeF4wpw7w9k", "https://www.youtube.com/watch?v=2NFwY15tRtA&t=9s", "https://www.youtube.com/watch?v=5dRramZVu2Q&t=48s"]
    for link in youtube_links:
        download_youtube_video(link)
        video_file = os.path.join("Videos", f"{YouTube(link).title}.mp4")
        split_video_into_frames(video_file)
    
        
if __name__ == "__main__":
    main()
