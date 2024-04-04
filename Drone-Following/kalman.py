import cv2
from filterpy.kalman import KalmanFilter
import numpy as np
from ultralytics import YOLO
import os

# Function to draw trajectories on the frame
def draw_trajectories(frame, trajectory, color):
    for i in range(1, len(trajectory)):
        if np.linalg.norm(np.array(trajectory[i]) - np.array(trajectory[i-1])) < 14:
            cv2.line(frame, (int(trajectory[i-1][0]), int(trajectory[i-1][1])), (int(trajectory[i][0]), int(trajectory[i][1])), color, 2)

# Initialize YOLO model with custom trained model
model = YOLO("Norsang_Model.pt")
model.overrides['conf'] = 0.25
model.overrides['iou'] = 0.45
model.overrides['agnostic_nms'] = False
model.overrides['max_det'] = 3
names = model.names

# Initialize Kalman filters 
cyclist_kf = KalmanFilter(dim_x=4, dim_z=2)
vehicle_kf = KalmanFilter(dim_x=4, dim_z=2)

# Configure Kalman filters
cyclist_kf.F = vehicle_kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
cyclist_kf.H = vehicle_kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
cyclist_kf.P = vehicle_kf.P = np.eye(4) * 1000
cyclist_kf.Q = vehicle_kf.Q = np.eye(4) * 0.01
cyclist_kf.R = vehicle_kf.R = np.eye(2) * 0.1

# List of video file paths
video_paths = [ "./Videos/Cyclist and vehicle Tracking - 1.mp4","./Videos/Cyclist and vehicle tracking - 2.mp4", "./Videos/Drone Tracking Video.mp4"]

# Process each video
for idx, video_path in enumerate(video_paths):
    cap = cv2.VideoCapture(video_path)
    
    # Setup video writer for output
    output_dir = "./kalman_video" 
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"kalman{idx+1}.mp4") 
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 640))
    
    cyclist_trajectory, vehicle_trajectory = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 640))
        results = model(frame)

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    if box.conf.cpu().numpy()[0] <= 0.40:
                        continue
                    class_index = int(box.cls)
                    class_name = names[class_index]
                    bbox_xyxy = box.xyxy.cpu().numpy()
                    bbox_center = ((bbox_xyxy[0][0] + bbox_xyxy[0][2]) / 2, (bbox_xyxy[0][1] + bbox_xyxy[0][3]) / 2)
                    
                    if class_name == 'cyclist':
                        cyclist_trajectory.append(bbox_center)
                        cyclist_kf.predict()
                        cyclist_kf.update(np.array(bbox_center)[:, np.newaxis])
                    else:
                        vehicle_trajectory.append(bbox_center)
                        vehicle_kf.predict()
                        vehicle_kf.update(np.array(bbox_center)[:, np.newaxis])
                        
                    cv2.rectangle(frame, (int(bbox_xyxy[0][0]), int(bbox_xyxy[0][1])), (int(bbox_xyxy[0][2]), int(bbox_xyxy[0][3])), (0, 255, 0) if class_name == "cyclist" else (0, 0, 255), 1)
                    cv2.putText(frame, class_name, (int(bbox_xyxy[0][0]), int(bbox_xyxy[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if class_name == "cyclist" else (0, 0, 255), 2)

        cyclist_trajectory = cyclist_trajectory[-50:]
        vehicle_trajectory = vehicle_trajectory[-50:]
        draw_trajectories(frame, cyclist_trajectory, (0, 255, 0))
        draw_trajectories(frame, vehicle_trajectory, (0, 0, 255))

        out.write(frame)

    cap.release()
    out.release()

cv2.destroyAllWindows()
