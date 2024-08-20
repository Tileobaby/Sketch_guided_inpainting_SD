import os
import shutil

# Define the source and destination directories
source_dir = 'image_real'
destination_dir = 'image_real_more'

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Loop through the range of x values
for x in range(1, 1001):
    # Format the source filename
    source_filename = f"{x:08d}.jpg"
    
    # Define the full path of the source file
    source_filepath = os.path.join(source_dir, source_filename)
    
    # Check if the source file exists
    if os.path.exists(source_filepath):
        # Loop through the range of y values
        for y in range(1, 6):
            # Format the destination filename
            destination_filename = f"{x:08d}_{y:02d}.jpg"
            
            # Define the full path of the destination file
            destination_filepath = os.path.join(destination_dir, destination_filename)
            
            # Copy and rename the file to the destination directory
            shutil.copy(source_filepath, destination_filepath)
            print(f"Copied {source_filepath} to {destination_filepath}")
    else:
        print(f"File {source_filepath} does not exist")

print("Copying completed.")