#!/bin/bash

# Directory to search for the largest file
DIRECTORY="log/."

# Function to get the largest file in the directory
get_largest_file() {
    find "$DIRECTORY" -type f -exec ls -lS {} + | tail -n 1 | awk '{print $NF}'
}

# Main loop
while true; do
    # Get the largest file
    FILE=$(get_largest_file)

    # Check if the file exists
    if [ -f "$FILE" ]; then
        # Print the last line of the largest file
        tail -n 1 "$FILE"

        # Wait for 1 second before updating
        sleep 1
    else
        echo "No files found in the directory."
        exit 1
    fi
done
