#!/bin/bash

# Define folder names
input_folder="./datasets/LIVE_NFLX_Plus/assets_mp4_individual"
output_folder="./datasets/LIVE_NFLX_Plus/assets_mp4_individual_no_sound"

# Create the output folder if it doesn't exist
if [ ! -d "$output_folder" ]; then
    mkdir -p "$output_folder"
    echo "Folder '$output_folder' created successfully"
else
    echo "Folder '$output_folder' already exists"
fi

# Process all MP4 files in the input folder
for input_file in "$input_folder"/*.mp4; do
    # Extract filename without extension
    filename=$(basename -- "$input_file")
    
    # Define output file path
    output_file="$output_folder/$filename"

    # Remove audio using ffmpeg
    ffmpeg -i "$input_file" -c copy -an "$output_file"

    echo "Processed: $input_file -> $output_file"
done

echo "All videos processed successfully."


