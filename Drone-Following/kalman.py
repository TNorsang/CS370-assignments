import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO
import os

def initialize_model():
    model = YOLO("Norsang_Model.pt")
    model.overrides.update({'conf': 0.25, 'iou': 0.45, 'agnostic_nms': False, 'max_det': 3})
    return model

def initialize_kalman_filters():
    filters = {}
    for obj_type in ['cyclist', 'vehicle']:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        kf.P *= 1000
        kf.Q = np.eye(4) * 0.01
        kf.R = np.eye(2) * 0.1
        filters[obj_type] = kf
    return filters

def process_frame(frame, model, filters, trajectories):
    frame = cv2.resize(frame, (640, 640))
    results = model(frame)

    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                if box.conf.cpu().numpy()[0] <= 0.40:
                    continue
                class_index = int(box.cls)
                class_name = model.names[class_index]
                bbox_xyxy = box.xyxy.cpu().numpy()  
                bbox_center = ((bbox_xyxy[0][0] + bbox_xyxy[0][2]) / 2, (bbox_xyxy[0][1] + bbox_xyxy[0][3]) / 2)
                
                obj_type = 'cyclist' if class_name == 'cyclist' else 'vehicle'
                kf = filters[obj_type]
                trajectories[obj_type].append(bbox_center)
                
                position_array = np.array(bbox_center)[:, np.newaxis]
                kf.predict()
                kf.update(position_array)

                draw_detections(frame, box, class_name)

    draw_trajectories(frame, trajectories)

def draw_detections(frame, box, class_name):
    x1, y1, x2, y2 = box.xyxy[0]
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0) if class_name == "cyclist" else (0, 0, 255), 1)
    label = f"{class_name}"
    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def draw_trajectories(frame, trajectories, dist_threshold=14, max_points=50):
    for obj_type, color in [('cyclist', (0, 255, 0)), ('vehicle', (0, 0, 255))]:
        trajectory = trajectories[obj_type][-max_points:]
        if len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                start_point = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
                end_point = (int(trajectory[i][0]), int(trajectory[i][1]))
                if np.linalg.norm(np.array(start_point) - np.array(end_point)) < dist_threshold:
                    cv2.line(frame, start_point, end_point, color, 2)

def process_videos(video_paths, model, filters):
    download_path = "./kalman_filter"
    os.makedirs(download_path, exist_ok=True)
    trajectories = {'cyclist': [], 'vehicle': []}

    for idx, video_path in enumerate(video_paths):
        cap = cv2.VideoCapture(video_path)
        output_filename = os.path.join(download_path, f"output_video_{idx}.mp4")
        out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 640))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            process_frame(frame, model, filters, trajectories)
            out.write(frame)

        cap.release()
        out.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_paths = ["./Videos/Cyclist and vehicle Tracking - 1.mp4", "./Videos/Cyclist and vehicle tracking - 2.mp4", "./Videos/Drone Tracking Video.mp4"]
    model = initialize_model()
    filters = initialize_kalman_filters()
    process_videos(video_paths, model, filters)
