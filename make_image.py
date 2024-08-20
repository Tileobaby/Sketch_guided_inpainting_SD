import os
import shutil

# Define the source and destination directories
source_dir = 'sketch_rendered/width-3'
destination_dir = 'real_sketch_31'

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Loop through the range of x values
for x in range(1, 1001):
    # Format the filenames
    old_filename = f"{x:08d}_01.png"
    new_filename = f"{x:08d}.png"
    
    # Define full file paths
    old_filepath = os.path.join(source_dir, old_filename)
    new_filepath = os.path.join(destination_dir, new_filename)
    
    # Check if the file exists in the source directory
    if os.path.exists(old_filepath):
        # Copy and rename the file to the destination directory
        shutil.copy(old_filepath, new_filepath)
        print(f"Copied {old_filepath} to {new_filepath}")
    else:
        print(f"File {old_filepath} does not exist")

print("Copying completed.")