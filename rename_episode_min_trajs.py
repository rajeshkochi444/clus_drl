import os

def rename_files_in_directory(directory):
    # Get all file names in the directory
    files = os.listdir(directory)

    # Filter out directories and hidden files
    files = [f for f in files if os.path.isfile(os.path.join(directory, f)) and not f.startswith('.')]

    # Sort files for consistent ordering
    files.sort()
    print(files)

    # Rename each file
    for index, file in enumerate(files):
        # Split the file name and extension
        file_name, file_extension = os.path.splitext(file)
        
        # Create the new file name
        #new_file_name = f"{index}_{file_name}{file_extension}"
        new_file_name = f"{index}{file_extension}"
        
        # Create the full path for the old and new file names
        old_file_path = os.path.join(directory, file)
        new_file_path = os.path.join(directory, new_file_name)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed '{file}' to '{new_file_name}'")

# Usage
# Replace 'your_directory_path' with the path of your folder
rename_files_in_directory('result_clusgym_ver50_expt1/episode_min/')
