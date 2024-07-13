 
#!/bin/bash

# Check if a program name was provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <program_name.py>"
  exit 1
fi

# Name of the conda environment you want to use
CONDA_ENV="p312"  # Replace with the name of your conda environment

# Path to the script or program you want to run
PROGRAM_PATH="$1"

# Number of times to launch the program
NUM_LAUNCHES=19

# Activate the conda environment
source ~/anaconda3/etc/profile.d/conda.sh  # Adjust the path to your conda.sh script if needed
conda activate $CONDA_ENV

# Launch the program multiple times
for ((i=1; i<=NUM_LAUNCHES; i++))
do
  nohup python $PROGRAM_PATH &  # Use 'python3' if needed
  echo "Launched $PROGRAM_PATH $i times."
  sleep 2  # Wait for 2 seconds before launching the next instance
done
