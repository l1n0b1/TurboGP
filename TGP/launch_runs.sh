 
#!/bin/bash

# Check if a program name was provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <program_name.py> <times to run>"
  exit 1
fi

# Name of the conda environment you want to use
CONDA_ENV="p312"  # Replace with the name of your conda environment

# Path to the script or program you want to run
PROGRAM_PATH="$1"

# Number of times to launch the program
second_arg="$2"
NUM_LAUNCHES=$((second_arg))

# Create the log folder if it doesn't exist
LOG_DIR="log"
mkdir -p $LOG_DIR

# Activate the conda environment
source ~/anaconda3/etc/profile.d/conda.sh  # Adjust the path to your conda.sh script if needed
conda activate $CONDA_ENV

# Launch the program multiple times
for ((i=1; i<=NUM_LAUNCHES; i++))
do

  # Define a unique output file for nohup in the log folder
  LOG_FILE="$LOG_DIR/nohup_${PROGRAM_PATH%.*}_$i.out"

  # Run the program with nohup and redirect output to the unique log file
  nohup python $PROGRAM_PATH > "$LOG_FILE" 2>&1 &

  echo "Launched $PROGRAM_PATH $i times."
  sleep 5  # Wait for 5 seconds before launching the next instance
done
