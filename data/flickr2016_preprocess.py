"""
@SUN, Haoran 2024/11/22 https://github.com/Tsuredur

This script checks if the image IDs listed in an order file are present in a specified image folder. 
It can be used to filter Flickr2016 by extracting images for train, test, and validation sets
based on their IDs in the `order` file.
"""

import os

order_file_path = '/PATH/TO/ORDER'
image_folder_path = '/PATH/TO/IMAGE'

# Check if exists
if not os.path.exists(image_folder_path):
    print(f"Error: Directory {image_folder_path} does not exist.")
    exit(1)

"""
Read all image IDs from the order file
Each line in the order file is expected to contain one image ID
"""
with open(order_file_path, 'r') as order_file:
    order_ids = [line.strip() for line in order_file.readlines()]

image_files = os.listdir(image_folder_path)

# Compare IDs with filenames
count = 0
for order_id in order_ids:
    if order_id in image_files:
        count += 1
    else:
        # Print a message if not found
        print(f"Image ID {order_id} not found in folder.")

print(f"Total number of matching images: {count}")

