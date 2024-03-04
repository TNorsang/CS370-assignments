Video Search : Object Detector
Norsang Nyandak Assignment1 Date: 2/28/2024

Running it locally:
Step 1 : Create a virtual environment for python
Command: **python3 -m venv myenv**

Step 2: Activate the environment
Command: **source myenv/bin/activate**

Step 3: Install all modules
Command: **pip install -r requirements.txt**

Two different paths for storing the csv and other needed data 
1) Local PC path 
2) Docker path
It is crucial to select the path you want to run in otherwise it would not work. These paths are commented out in the ObjectDetector.py file.


Running it through Docker:
Build Command : **docker build -t object_detector .**
Run Command : **docker run object_detector**
Make sure the path of the "output_path" is updated to this:
**output_path = "/ObjectDetector"**



Postgres through Docker
Step 1: docker exec -it my_postgres bash
Step 2: psql -U postgres
then create your ursername and password and database etc.
My database = object_detector
if you run the file 'postgres.py'
it will succesfully create the table.
Access and view database in Docker.
Command : **psql -U norsang -d object_detector**
Command : **SELECT * FROM object_detector_table;**
