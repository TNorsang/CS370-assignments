import torch
import numpy as np
import cv2
from pytube import YouTube

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def download_youtube_video(youtube_link):
    yt = YouTube(youtube_link)
    video_stream = yt.streams.first()


def main():
    youtube_links = ["https://www.youtube.com/watch?v=WeF4wpw7w9k", "https://www.youtube.com/watch?v=2NFwY15tRtA&t=9s", "https://www.youtube.com/watch?v=5dRramZVu2Q&t=48s"]

    for i in youtube_links:
        print(f"Youtube Links are: {i}")
        

if __name__ == "__main__":
    main()

# def detect_objects(image_path):

#     img = cv2.