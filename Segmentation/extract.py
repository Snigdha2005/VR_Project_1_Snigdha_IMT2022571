import os
import zipfile

# Path to the specific zip file and output folder
input_file = os.path.expanduser('~/MSFD.zip')
output_folder = os.path.expanduser('~/VR_project_1')

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

def extract_file(file_path, output_path):
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        print(f"Extracted: {file_path} to {output_path}")
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")

def main():
    if os.path.isfile(input_file) and input_file.endswith('.zip'):
        extract_file(input_file, output_folder)
    else:
        print("File not found or not a zip file.")

if __name__ == '__main__':
    main()

