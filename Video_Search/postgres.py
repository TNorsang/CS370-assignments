import pandas as pd
import psycopg2
from sqlalchemy import create_engine

db_config = {
    'dbname': 'object_detector',
    'user': 'norsang',
    'password': 'password',
    'host': 'localhost',
    'port': 5432 
}

def create_table(conn, table_name):
    try:
        cursor = conn.cursor()
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (ID SERIAL PRIMARY KEY, frameNum INTEGER, timestamp VARCHAR(20), detectedObjId INTEGER, detectedObjClass VARCHAR(50), confidence FLOAT, bbox_info VARCHAR(100), encoded_vector BYTEA);")
        conn.commit()
        cursor.close()
    except psycopg2.Error as e:
        print(f"Error creating table: {e}")

def upload_csv_to_postgres(csv_file_path, db_config):
    conn_string = f"dbname='{db_config['dbname']}' user='{db_config['user']}' host='{db_config['host']}' password='{db_config['password']}' port='{db_config['port']}'"
    
    try:
        conn = psycopg2.connect(conn_string)
        create_table(conn, "object_detector_table")  

        # Read CSV file
        df = pd.read_csv(csv_file_path)

        # Get unique combinations of columns
        unique_combinations = df.drop_duplicates().values.tolist()

        for i, combination in enumerate(unique_combinations):
            table_name = f"object_detector_table_{i}"
            create_table(conn, table_name)
            # Insert data into table
            cursor = conn.cursor()
            for row in df[df.apply(lambda x: x.tolist() == combination, axis=1)].values.tolist():
                cursor.execute(f"INSERT INTO {table_name} (frameNum, timestamp, detectedObjId, detectedObjClass, confidence, bbox_info, encoded_vector) VALUES (%s, %s, %s, %s, %s, %s, %s)", row[1:])
            conn.commit()
            cursor.close()

        print("Tables created and data uploaded successfully.")
        conn.close()
    except psycopg2.Error as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # CSV file path for windows
    csv_file_path = 'C:/Users/theno/OneDrive/Documents/Spring 2024/CS370-102 Introduction to Artificial Intelligence/CS370-assignments/Video_Search/Main/CSV/all_videos_results.csv'

    # Docker path
    # csv_file_path = "/ObjectDetector"
    
    upload_csv_to_postgres(csv_file_path, db_config)
