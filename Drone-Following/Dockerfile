# Use the Python base image
FROM python:latest

# Set the working directory inside the container
WORKDIR /DroneApp

# Copy the project files into the container
COPY . .

# Install required system dependencies for OpenCV, OpenGL, and HDF5
RUN apt-get update && apt-get install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev libhdf5-dev

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install h5py separately due to potential build dependencies
RUN pip install --no-cache-dir h5py

# Command to run the application
CMD ["python", "DroneFollowMe.py"]
