"""
Harry Potter Invisibility Cloak using OpenCV and Python.

How it works:
1. Capture and store the background frame when the subject is not in front of the camera.
2. Detect the chosen cloth color in real-time by converting frames from BGR to HSV color space.
3. Generate a binary mask for the detected color and apply morphological operations to remove noise.
4. Segment the background corresponding to the mask area.
5. Invert the mask and segment the current frame (keeping everything except the cloak).
6. Blend both images together to produce the magical invisibility effect.

Key Controls:
  'b' - Recapture background frame
  '1' - Select Red cloak mode
  '2' - Select Green cloak mode
  '3' - Select Blue cloak mode
  'c' - Click on video to auto-calibrate cloak color
  't' - Open / Close HSV trackbars for fine-tuning
  'm' - Toggle Split / Mask view for debugging
  's' - Save snapshot
  'q' / Esc - Quit
"""

import cv2
import numpy as np
import time

# Preset HSV ranges for popular cloak colors
COLOR_PRESETS = {
    "red": {
        "ranges": [
            (np.array([0, 120, 70]), np.array([10, 255, 255])),
            (np.array([170, 120, 70]), np.array([180, 255, 255]))
        ],
        "name": "Red"
    },
    "green": {
        "ranges": [
            (np.array([35, 70, 50]), np.array([85, 255, 255]))
        ],
        "name": "Green"
    },
    "blue": {
        "ranges": [
            (np.array([95, 100, 50]), np.array([130, 255, 255]))
        ],
        "name": "Blue"
    }
}

# Calibration state for mouse clicks
clicked_hsv = None

def mouse_callback(event, x, y, flags, param):
    """Mouse callback to capture clicked pixel color in HSV."""
    global clicked_hsv
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_hsv = param.get("hsv_frame")
        if frame_hsv is not None and 0 <= y < frame_hsv.shape[0] and 0 <= x < frame_hsv.shape[1]:
            clicked_hsv = frame_hsv[y, x]
            print(f"[Calibrate] Clicked at ({x}, {y}) - HSV: {clicked_hsv}")

def nothing(x):
    """Empty callback for trackbars."""
    pass

def create_trackbars():
    """Create window and sliders for real-time HSV threshold adjustments."""
    trackbar_win = "HSV Calibration"
    cv2.namedWindow(trackbar_win, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Lower H", trackbar_win, 0, 180, nothing)
    cv2.createTrackbar("Upper H", trackbar_win, 10, 180, nothing)
    cv2.createTrackbar("Lower S", trackbar_win, 120, 255, nothing)
    cv2.createTrackbar("Upper S", trackbar_win, 255, 255, nothing)
    cv2.createTrackbar("Lower V", trackbar_win, 70, 255, nothing)
    cv2.createTrackbar("Upper V", trackbar_win, 255, 255, nothing)
    return trackbar_win

def get_trackbar_values(win_name):
    """Retrieve current values from HSV trackbars."""
    l_h = cv2.getTrackbarPos("Lower H", win_name)
    u_h = cv2.getTrackbarPos("Upper H", win_name)
    l_s = cv2.getTrackbarPos("Lower S", win_name)
    u_s = cv2.getTrackbarPos("Upper S", win_name)
    l_v = cv2.getTrackbarPos("Lower V", win_name)
    u_v = cv2.getTrackbarPos("Upper V", win_name)
    return np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v])

def capture_background(cap, num_frames=30, countdown_sec=3):
    """
    Captures the static background before the user steps into the frame.
    Takes the average or latest clear frame after a brief warm-up.
    """
    print("\n[Notice] Please step out of the camera view. Capturing background...")
    
    # Warm-up camera & show countdown
    start_time = time.time()
    while time.time() - start_time < countdown_sec:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        remaining = int(countdown_sec - (time.time() - start_time)) + 1
        display = frame.copy()
        
        cv2.putText(
            display,
            f"Step out of frame! Capturing in {remaining}s...",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 140, 255),
            2,
            cv2.LINE_AA
        )
        cv2.imshow("Invisibility Cloak", display)
        if cv2.waitKey(30) & 0xFF == 27:  # Esc
            return None

    # Collect frames for a clean, stable background snapshot
    bg_frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            bg_frames.append(frame.astype(np.float32))
        cv2.waitKey(20)

    if not bg_frames:
        raise RuntimeError("Failed to capture background from webcam.")

    # Average frames to reduce camera sensor noise
    avg_bg = np.mean(bg_frames, axis=0).astype(np.uint8)
    print("[Success] Background captured successfully!\n")
    return avg_bg

def build_mask(hsv, color_mode, custom_ranges, trackbar_mode, trackbar_win):
    """
    Creates the binary mask for the cloak based on current mode and HSV values.
    """
    if trackbar_mode and trackbar_win:
        lower, upper = get_trackbar_values(trackbar_win)
        mask = cv2.inRange(hsv, lower, upper)
    elif custom_ranges is not None:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in custom_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    else:
        preset = COLOR_PRESETS.get(color_mode, COLOR_PRESETS["red"])
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in preset["ranges"]:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    # Morphological transformations to refine the mask
    # 1. Opening: removes small noise specks
    kernel_open = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # 2. Dilation: fills in pinholes inside the cloak area
    kernel_dilate = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_dilate, iterations=1)

    return mask

def apply_invisibility(frame, background, mask):
    """
    Blends the background into the current frame using smooth alpha feathering.
    """
    # Create smooth edge transition using Gaussian Blur
    mask_blurred = cv2.GaussianBlur(mask, (7, 7), 0)
    
    # Normalize mask to 0.0 - 1.0 float for seamless blending
    alpha = (mask_blurred.astype(np.float32) / 255.0)[:, :, np.newaxis]

    # Cloak region = background, Non-cloak region = current frame
    foreground = frame.astype(np.float32)
    bg = background.astype(np.float32)

    blended = (alpha * bg) + ((1.0 - alpha) * foreground)
    return np.clip(blended, 0, 255).astype(np.uint8)

def main():
    global clicked_hsv

    # Initialize video capture from default webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Error] Could not open webcam. Please verify your camera is connected.")
        return

    # Allow camera sensor to initialize
    time.sleep(1.0)

    # Capture initial background
    background = capture_background(cap)
    if background is None:
        print("[Notice] Background capture cancelled by user.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # Window setup
    main_win = "Invisibility Cloak"
    cv2.namedWindow(main_win, cv2.WINDOW_AUTOSIZE)
    
    # Context dictionary passed to mouse callback
    mouse_param = {"hsv_frame": None}
    cv2.setMouseCallback(main_win, mouse_callback, mouse_param)

    # State variables
    color_mode = "red"
    custom_ranges = None
    trackbar_mode = False
    trackbar_win = None
    debug_mode = False

    print("=" * 60)
    print(" INVISIBILITY CLOAK RUNNING")
    print("=" * 60)
    print("Controls:")
    print("  'b'       : Recapture background")
    print("  '1'       : Red cloak preset")
    print("  '2'       : Green cloak preset")
    print("  '3'       : Blue cloak preset")
    print("  'c'       : Click on your cloth to auto-calibrate")
    print("  't'       : Toggle real-time HSV sliders window")
    print("  'm'       : Toggle debug / split mask view")
    print("  's'       : Save screenshot")
    print("  'q' / Esc : Quit")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to receive frame from webcam.")
            break

        # Flip horizontally for natural mirror feel
        frame = cv2.flip(frame, 1)

        # Ensure background matches current frame dimensions
        if background.shape != frame.shape:
            background = cv2.resize(background, (frame.shape[1], frame.shape[0]))

        # Convert frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mouse_param["hsv_frame"] = hsv

        # Check if user clicked to calibrate a color
        if clicked_hsv is not None:
            h, s, v = int(clicked_hsv[0]), int(clicked_hsv[1]), int(clicked_hsv[2])
            print(f"[Calibrate] Cloth calibrated: H={h}, S={s}, V={v}")
            # If hue is close to red boundaries (0 or 180), use 2 ranges
            if h < 15 or h > 165:
                custom_ranges = [
                    (np.array([0, max(50, s - 60), max(50, v - 60)]), np.array([15, 255, 255])),
                    (np.array([165, max(50, s - 60), max(50, v - 60)]), np.array([180, 255, 255]))
                ]
            else:
                l_h = max(0, h - 15)
                u_h = min(180, h + 15)
                l_s = max(50, s - 60)
                u_s = 255
                l_v = max(50, v - 60)
                u_v = 255
                custom_ranges = [(np.array([l_h, l_s, l_v]), np.array([u_h, u_s, u_v]))]
            
            color_mode = "custom (click-calibrated)"
            clicked_hsv = None

        # Build mask for the cloak
        mask = build_mask(hsv, color_mode, custom_ranges, trackbar_mode, trackbar_win)

        # Apply invisibility blend
        output = apply_invisibility(frame, background, mask)

        # Overlay HUD info on output
        hud_text = f"Mode: {color_mode.capitalize()} | 'b': Recapture BG | 'c': Calibrate | 'm': Mask view"
        cv2.putText(
            output,
            hud_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Debug mode: show original, mask, and output side-by-side
        if debug_mode:
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack([
                cv2.resize(frame, (320, 240)),
                cv2.resize(mask_3ch, (320, 240)),
                cv2.resize(output, (320, 240))
            ])
            cv2.imshow("Debug View [Original | Mask | Invisibility]", combined)
        else:
            if cv2.getWindowProperty("Debug View [Original | Mask | Invisibility]", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("Debug View [Original | Mask | Invisibility]")

        cv2.imshow(main_win, output)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q'), ord('Q')):  # Esc or 'q'
            break
        elif key in (ord('b'), ord('B')):
            new_bg = capture_background(cap, countdown_sec=2)
            if new_bg is not None:
                background = new_bg
        elif key == ord('1'):
            color_mode = "red"
            custom_ranges = None
            print("[Mode] Switched to Red cloak preset.")
        elif key == ord('2'):
            color_mode = "green"
            custom_ranges = None
            print("[Mode] Switched to Green cloak preset.")
        elif key == ord('3'):
            color_mode = "blue"
            custom_ranges = None
            print("[Mode] Switched to Blue cloak preset.")
        elif key in (ord('t'), ord('T')):
            trackbar_mode = not trackbar_mode
            if trackbar_mode:
                trackbar_win = create_trackbars()
                print("[Trackbars] Calibration sliders opened.")
            else:
                if trackbar_win:
                    cv2.destroyWindow(trackbar_win)
                    trackbar_win = None
                print("[Trackbars] Calibration sliders closed.")
        elif key in (ord('m'), ord('M')):
            debug_mode = not debug_mode
            print(f"[Debug] Mask view {'enabled' if debug_mode else 'disabled'}.")
        elif key in (ord('s'), ord('S')):
            timestamp = int(time.time())
            filename = f"invisibility_snapshot_{timestamp}.png"
            cv2.imwrite(filename, output)
            print(f"[Snapshot] Saved to {filename}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[Shutdown] Closed camera and windows successfully.")

if __name__ == "__main__":
    main()
