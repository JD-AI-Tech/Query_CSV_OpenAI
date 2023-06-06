import os
from file_clean_up import delete_processed_files

def create_directory(directory_path):
    try:
        if not os.path.exists(directory_path):
           os.makedirs(directory_path)
           print(f"Directory acreated: {directory_path}")
        else:
            print(f"Directory already exists: {directory_path}")
            delete_processed_files(directory_path)
    except OSError as e:
        print(f"Error creating directory: {directory_path}")
        print(e)

# Example usage:
if __name__ == "__main__":
    create_directory("./log")
