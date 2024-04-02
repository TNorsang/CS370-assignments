# Use the Python base image
FROM python:latest

# Set the working directory inside the container
WORKDIR /DroneApp

# Copy the project files into the container
COPY . .

# Install required system dependencies for OpenCV and OpenGL
RUN apt-get update && apt-get install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev

# Install required Python packages
RUN pip install --no-cache-dir pytube opencv-python torch torchvision pandas keras tensorflow ultralytics

# Command to run the application
CMD ["python", "DroneFollowMe.py"]
