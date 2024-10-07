from filelock import FileLock, Timeout
import time

import numpy as np

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]


def append_to_csv_row(file_path, value):
    ''' This function appends a row (e.g., params, testing fitness results, elapsed time, etc)
        to text file. It requires as input the file name and the string to store.'''

    # Define the path to the text file and the lock file
    lock_path = file_path + '.lock'

    # Create a FileLock object
    lock = FileLock(lock_path, timeout=10)  # Timeout is optional

    try:
        with lock:
            # Open the file in append mode and write the float value
            with open(file_path, 'a') as file:
                file.write(f"{value}\n")
            #print(f"Successfully wrote {value} to {file_path}")
    except Timeout:
        print("Could not acquire the lock. Try again later.")
    except Exception as e:
        print(f"An error occurred: {e}")

