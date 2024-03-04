from pytube import YouTube
import cv2  
import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from PIL import Image
import pandas as pd
import random
import numpy as np
import datetime
from keras import layers
from keras.preprocessing.image import img_to_array, array_to_img
import keras


coco_class = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush"
}

def safe_filename(filename):
    keepcharacters = (' ', '.', '_', '-')
    return "".join(c for c in filename if c.isalnum() or c in keepcharacters).rstrip()

def download_video(video_url, output_path):
    try:
        yt = YouTube(video_url)
        yt.bypass_age_gate()
        caption = yt.captions["a.en"]
        if caption == {}:
            caption = yt.captions["en"]
            
        stream = yt.streams.get_highest_resolution()
        safe_title = safe_filename(stream.title)
        caption_title = safe_filename(stream.title + " Caption")
        video_folder = os.path.join(output_path, safe_title)
        full_path = os.path.join(video_folder, safe_title + ".mp4")
        caption_path = os.path.join(video_folder, caption_title + ".srt") 

      
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
        
     
        stream.download(video_folder, safe_title + ".mp4")
    
        if caption:
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(caption.xml_captions)
            print(f"Caption downloaded successfully to: {caption_path}")
        else:
            print("No caption was found.")
            
        print(f"Video downloaded successfully to: {full_path}")
        return video_folder, full_path  
    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        return None

def load_model():
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    return model

def detect_objects(model, frame_path, confidence_threshold=0.7):
    img = Image.open(frame_path).convert("RGB")
    img_tensor = F.to_tensor(img)
    with torch.no_grad():
        prediction = model([img_tensor])[0]

    pred_scores = prediction['scores']
    high_conf_pred_indices = [i for i, score in enumerate(pred_scores) if score > confidence_threshold]
    high_conf_predictions = {
        'labels': [coco_class.get(i, 'unknown') for i in prediction['labels'][high_conf_pred_indices].tolist()],
        'scores': prediction['scores'][high_conf_pred_indices].tolist(),
        'boxes': prediction['boxes'][high_conf_pred_indices].tolist()
    }

    return high_conf_predictions

def draw_bounding_boxes(frame_path, predictions):
    image = cv2.imread(frame_path)
    if 'boxes' in predictions:
        for i, box in enumerate(predictions['boxes']):
            box = [int(coord) for coord in box]  
            start_point = tuple(box[:2])
            end_point = tuple(box[2:])
            color = (0, 0, 255) 
            thickness = 2
            cv2.rectangle(image, start_point, end_point, color, thickness)
            label = f"{predictions['labels'][i]}: {predictions['scores'][i]:.2f}"
            cv2.putText(image, label, (start_point[0], start_point[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(frame_path.replace('.jpg', '_detected.jpg'), image)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
 
def preprocess_detected_objects(image, predictions, resize_dim, output_folder, frame_id):
    for i, box in enumerate(predictions['boxes']):
        box = [int(coord) for coord in box]  
        cropped_img = image.crop(box)  
        resized_img = cropped_img.resize(resize_dim)  
        
       
        image_filename = f"object_{frame_id}_{i}.png"
        image_path = os.path.join(output_folder, image_filename)
        
      
        resized_img.save(image_path)

 
def build_autoencoder(input_shape=(64, 64, 3)):
    input_img = keras.Input(shape=input_shape)
    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    # Decoder
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

def encode_objects_with_autoencoder(input_folder, autoencoder_model):
    encoded_vectors = [] 
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array = img.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)  
            encoded_representation = autoencoder_model.predict(img_array)
            encoded_vectors.append(encoded_representation.flatten())
    return encoded_vectors


                
def extract_frames_and_detect_objects(video_file_path, model, vidId, interval, output_folder, video_folder_name, autoencoder_model, all_results):
    vidcap = cv2.VideoCapture(video_file_path)
    success, image = vidcap.read()
    count = 0
    saved_frames = 0
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames_dir = os.path.join(output_folder, video_folder_name, 'FramesDetected')
    objects_dir = os.path.join(output_folder, video_folder_name, 'ObjectsDetected')
    ensure_dir(frames_dir)
    ensure_dir(objects_dir)
    results = []

    while success and saved_frames < 15:
        if count % (int(fps) * interval) == 0:
            frame_path = os.path.join(frames_dir, f"frame{saved_frames}.jpg")
            cv2.imwrite(frame_path, image)
            prediction = detect_objects(model, frame_path)
            if 'labels' in prediction:
                draw_bounding_boxes(frame_path, prediction)
                preprocess_detected_objects(Image.fromarray(image), prediction, resize_dim=(64, 64), output_folder=objects_dir, frame_id=saved_frames)
                
                encoded_vectors = encode_objects_with_autoencoder(objects_dir, autoencoder_model)

                for i, label in enumerate(prediction['labels']):
                    box_info = prediction['boxes'][i]
                    box_info = [int(coord) for coord in box_info]  
                    encoded_vector = encoded_vectors[i] if i < len(encoded_vectors) else []
                    results.append([
                        vidId,
                        saved_frames,
                        str(datetime.timedelta(seconds=int(count / fps))),
                        i,
                        label,
                        prediction['scores'][i],
                        box_info, 
                        list(encoded_vector)  
                    ])
                saved_frames += 1
        success, image = vidcap.read()
        count += 1

    all_results.extend(results)

    return all_results





if __name__ == "__main__":
    model = load_model()
    autoencoder_model = build_autoencoder()

    video_urls = [
        "https://www.youtube.com/watch?v=wbWRWeVe1XE",
        "https://www.youtube.com/watch?v=FlJoBhLnqko",
        "https://www.youtube.com/watch?v=Y-bVwPRy_no"
    ]

    output_path = "C:/Users/theno/OneDrive/Documents/Spring 2024/CS370-102 Introduction to Artificial Intelligence/CS370-assignments/Video_Search/Main"

    preprocessed_base_folder = os.path.join(output_path, "Preprocessed")
    ensure_dir(preprocessed_base_folder)

    all_results = []  

    for url in video_urls:
        video_folder, video_file_path = download_video(url, output_path)
        if video_folder and video_file_path:
            videoID = random.randrange(100)
            safe_title = os.path.basename(video_folder)
            preprocessed_folder = os.path.join(preprocessed_base_folder, safe_title, 'ObjectsDetected')
            ensure_dir(preprocessed_folder)

            extract_frames_and_detect_objects(video_file_path, model, videoID, 10, preprocessed_base_folder, safe_title, autoencoder_model, all_results)
    
    columns = ['vidId', 'frameNum', 'timestamp(H:MM:SS)', 'detectedObjId', 'detectedObjClass', 'confidence', 'bbox info', 'encoded vector']
    final_df = pd.DataFrame(all_results, columns=columns)
    
    csv_output_folder = os.path.join(output_path, "CSV")
    ensure_dir(csv_output_folder)
    csv_file_path = os.path.join(csv_output_folder, "all_videos_results.csv")
    final_df.to_csv(csv_file_path, index=False)
    print(f"All results saved to CSV at {csv_file_path}")











    
