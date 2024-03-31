import torch
import numpy as np
import cv2
from pytube import YouTube
import os

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def download_youtube_video(youtube_link, id):
    yt = YouTube(youtube_link)
    video_stream = yt.streams.first()
    # Specifying the name of the folder where the Videos are stored in
    # Creating the Videos Folder

    # videos_output_path = os.path.join("Videos", f"{yt.title}.mp4")
    # os.makedirs(video_output_folder, exist_ok=True)
    video_stream.download(output_path="Videos")


def main():
    youtube_links = ["https://www.youtube.com/watch?v=WeF4wpw7w9k", "https://www.youtube.com/watch?v=2NFwY15tRtA&t=9s", "https://www.youtube.com/watch?v=5dRramZVu2Q&t=48s"]
    j = 1
    for i in youtube_links:
        download_youtube_video(i,j)
        j+=1
    
        
if __name__ == "__main__":
    main()
