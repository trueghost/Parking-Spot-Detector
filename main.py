import cv2
import matplotlib.pyplot as plt
import numpy as np

# Import functions from the 'util' module
from util import get_parking_spots_bboxes, empty_or_not


def calc_diff(im1, im2):
    # Calculate the absolute difference in mean pixel values between two images
    return np.abs(np.mean(im1) - np.mean(im2))


# File paths for the mask image and the video
mask = './masks/mask_name'
video_path = './data/video_name'

# Read the mask image (a binary image representing parking spots)
mask = cv2.imread(mask, 0)

# Open the video capture object for the specified video file
cap = cv2.VideoCapture(video_path)

# Perform connected component analysis on the mask image to identify parking spots
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

# Extract the bounding boxes of parking spots from the connected components data
spots = get_parking_spots_bboxes(connected_components)

# Lists to store the status (empty or not) and differences for each parking spot
spots_status = [None for j in spots]
diffs = [None for j in spots]

# Initialize variables to keep track of frames and previous frame
previous_frame = None
frame_nmr = 0

# A flag variable to check if a frame is successfully read from the video
ret = True

# A variable to control how many frames to skip before processing the next frame
step = 30

# Start processing frames from the video
while ret:
    # Read the current frame from the video
    ret, frame = cap.read()

    # Check if the current frame number is a multiple of 'step' and if we have a previous frame
    if frame_nmr % step == 0 and previous_frame is not None:
        # For each parking spot, calculate the difference in pixel values between this frame and the previous frame
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot

            # Crop the current frame to the region corresponding to the parking spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            # Calculate the difference in pixel values between the cropped region and the corresponding region in the previous frame
            diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

        # Sort and print the differences in descending order
        print([diffs[j] for j in np.argsort(diffs)][::-1])

        # Uncomment the following lines to plot a histogram of the differences
        # plt.figure()
        # plt.hist([diffs[j] / np.amax(diffs) for j in np.argsort(diffs)][::-1], bins=20)
        # if frame_nmr == 300:
        #     plt.show()

    # Check if the current frame number is a multiple of 'step'
    if frame_nmr % step == 0:
        # If it's the first frame, process all spots; otherwise, select spots with significant differences
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
        for spot_indx in arr_:
            spot = spots[spot_indx]
            x1, y1, w, h = spot

            # Crop the current frame to the region corresponding to the parking spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            # Determine if the parking spot is empty or occupied
            spot_status = empty_or_not(spot_crop)

            # Update the status of the parking spot in the spots_status list
            spots_status[spot_indx] = spot_status

    # Update previous_frame with the current frame for the next iteration
    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    # Draw rectangles around each parking spot on the frame, colored green for empty spots and red for occupied spots
    for spot_indx, spot in enumerate(spots):
        spot_status = spots_status[spot_indx]
        x1, y1, w, h = spots[spot_indx]

        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    # Display the count of available spots at the top-left corner of the frame
    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))), (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame in a window named 'frame' using OpenCV
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)

    # Wait for a key press and check if 'q' is pressed to break the loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # Increment the frame number
    frame_nmr += 1

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()