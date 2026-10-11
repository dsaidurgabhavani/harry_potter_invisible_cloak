"""
Offline Demonstration of the Invisibility Cloak Pipeline.
Generates synthetic frames using OpenCV and executes the exact masking & blending
algorithm to output demonstration images without needing a physical webcam.
"""

import cv2
import numpy as np

def create_synthetic_background(width=640, height=480):
    """Creates a synthetic room background with wall, floor, poster, and table."""
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Wall (top half) - soft slate blue
    bg[:320, :] = [80, 60, 50]
    
    # Floor (bottom half) - dark wood brown
    bg[320:, :] = [20, 45, 95]
    
    # Wood floor planks
    for x in range(0, width, 80):
        cv2.line(bg, (x, 320), (x - 30, height), (15, 30, 65), 2)
        
    # Wall Poster / Picture Frame
    cv2.rectangle(bg, (240, 60), (400, 180), (30, 180, 220), 4) # Gold frame
    bg[64:176, 244:396] = [40, 40, 120] # Poster canvas
    cv2.putText(bg, "HOGWARTS", (260, 125), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    
    # Table on the side
    cv2.rectangle(bg, (40, 260), (160, 380), (35, 70, 130), -1)
    # Book on table
    cv2.rectangle(bg, (60, 240), (120, 260), (180, 50, 40), -1)

    return bg

def create_synthetic_person_with_cloak(bg, cloak_color_bgr=(20, 20, 220)):
    """Creates a frame with a person standing in the room holding a cloak."""
    frame = bg.copy()
    
    center_x, center_y = 320, 240
    
    # Head (skin tone)
    cv2.circle(frame, (center_x, 150), 35, (180, 200, 240), -1)
    # Hair
    cv2.ellipse(frame, (center_x, 135), (38, 25), 0, 180, 360, (20, 20, 25), -1)
    # Glasses
    cv2.circle(frame, (center_x - 12, 155), 10, (10, 10, 10), 2)
    cv2.circle(frame, (center_x + 12, 155), 10, (10, 10, 10), 2)
    cv2.line(frame, (center_x - 2, 155), (center_x + 2, 155), (10, 10, 10), 2)
    
    # Legs
    cv2.rectangle(frame, (center_x - 25, 380), (center_x - 5, 460), (40, 30, 30), -1)
    cv2.rectangle(frame, (center_x + 5, 380), (center_x + 25, 460), (40, 30, 30), -1)
    # Shoes
    cv2.rectangle(frame, (center_x - 30, 455), (center_x - 5, 470), (10, 10, 10), -1)
    cv2.rectangle(frame, (center_x + 5, 455), (center_x + 30, 470), (10, 10, 10), -1)

    # Cloak (covering body from shoulders to mid-thigh)
    pts = np.array([
        [center_x - 55, 190],
        [center_x + 55, 190],
        [center_x + 70, 380],
        [center_x - 70, 380]
    ], np.int32)
    cv2.fillPoly(frame, [pts], cloak_color_bgr)
    
    return frame

def run_invisibility_algorithm(frame, background):
    """Executes the exact invisibility cloak algorithm."""
    # 1. Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 2. Red color thresholding (handling both wraps of Red in HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 | mask2

    # 3. Morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8), iterations=1)

    # 4. Invert mask
    mask_inv = cv2.bitwise_not(mask)

    # 5. Segment cloak area from background and non-cloak area from current frame
    cloak_area = cv2.bitwise_and(background, background, mask=mask)
    non_cloak_area = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # 6. Smooth blend
    result = cv2.addWeighted(cloak_area, 1, non_cloak_area, 1, 0)
    
    return mask, result

def main():
    print("Generating synthetic test frames...")
    bg = create_synthetic_background()
    raw = create_synthetic_person_with_cloak(bg)
    
    print("Applying Invisibility Cloak Computer Vision algorithm...")
    mask, output = run_invisibility_algorithm(raw, bg)
    
    # Save individual demonstration images
    cv2.imwrite("demo_1_background.png", bg)
    cv2.imwrite("demo_2_raw_frame.png", raw)
    cv2.imwrite("demo_3_hsv_mask.png", mask)
    cv2.imwrite("demo_4_invisible_result.png", output)
    
    # Build 2x2 comparison grid
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Label images
    cv2.putText(bg, "1. Stored Background", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(raw, "2. Live Frame (Wearing Cloak)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(mask_bgr, "3. Detected HSV Mask", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(output, "4. Invisibility Illusion Result", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    top_row = np.hstack([bg, raw])
    bot_row = np.hstack([mask_bgr, output])
    comparison = np.vstack([top_row, bot_row])
    
    cv2.imwrite("demo_comparison_grid.png", comparison)
    print("\nDemonstration images successfully generated:")
    print("  - demo_1_background.png")
    print("  - demo_2_raw_frame.png")
    print("  - demo_3_hsv_mask.png")
    print("  - demo_4_invisible_result.png")
    print("  - demo_comparison_grid.png")

if __name__ == "__main__":
    main()
