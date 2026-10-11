"""
Simple Invisibility Cloak using OpenCV.
Minimal implementation demonstrating the core computer vision algorithm.
"""

import cv2
import numpy as np
import time

def main():
    cap = cv2.VideoCapture(0)
    time.sleep(2)  # Give camera time to warm up

    print("Step out of the frame! Capturing background in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    # 1. Capture the static background
    background = None
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            background = cv2.flip(frame, 1)

    if background is None:
        print("Error: Could not capture background.")
        cap.release()
        return

    print("Background captured! Put on your red cloth now. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # 2. Convert from BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 3. Detect red cloth (Red wraps around 0 and 180 in HSV)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        # Combine masks
        mask = mask1 | mask2

        # 4. Morphological operations to filter noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=1)

        # 5. Invert mask to get area where cloak is NOT present
        mask_inv = cv2.bitwise_not(mask)

        # 6. Segment background where cloth is present
        cloak_area = cv2.bitwise_and(background, background, mask=mask)

        # 7. Segment current frame where cloth is NOT present
        non_cloak_area = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # 8. Combine both to produce the illusion
        final_output = cv2.addWeighted(cloak_area, 1, non_cloak_area, 1, 0)

        cv2.imshow("Invisibility Cloak (Simple)", final_output)

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
