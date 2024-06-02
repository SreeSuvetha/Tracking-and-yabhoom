import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands object
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1
)

# Initialize grid and marker position
grid_size = 50
grid = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
marker_pos = [5, 10]  # Initial position (row, col)
marker_path = [tuple(marker_pos)]  # List to store the marker's path

# Time tracking variables
last_update_time = time.time()
update_interval = 0.001  # 0.5 second interval

# Variable to store the last detected gesture
last_detected_gesture = None

# Initialize obstacles
obstacles = [(20, 20), (20, 21), (20, 22), (30, 30), (30, 31), (31, 31), (30, 32), (31, 32), (32, 32), (33, 32), (33, 33)]  # Add more obstacles as needed
for obs in obstacles:
    grid[obs[0], obs[1]] = (0, 0, 255)  # Draw obstacle (red)

# Function to classify hand gestures
def count_fingers(landmarks):
    def is_finger_extended(tip, dip, pip, mcp):
        return tip.y < dip.y < pip.y < mcp.y

    fingers_extended = 0

    # Thumb
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    thumb_cmc = landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    if thumb_tip.x > thumb_ip.x and thumb_tip.y < thumb_mcp.y:
        fingers_extended += 1

    # Index finger
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_dip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    index_pip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_mcp = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    if is_finger_extended(index_tip, index_dip, index_pip, index_mcp):
        fingers_extended += 1

    # Middle finger
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_dip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    middle_pip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    middle_mcp = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    if is_finger_extended(middle_tip, middle_dip, middle_pip, middle_mcp):
        fingers_extended += 1

    # Ring finger
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_dip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
    ring_pip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    ring_mcp = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    if is_finger_extended(ring_tip, ring_dip, ring_pip, ring_mcp):
        fingers_extended += 1

    # Pinky finger
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_dip = landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
    pinky_pip = landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    pinky_mcp = landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    if is_finger_extended(pinky_tip, pinky_dip, pinky_pip, pinky_mcp):
        fingers_extended += 1

    return fingers_extended

# Function to update the marker position
def update_marker_position(finger_count):
    global marker_pos

    new_pos = marker_pos.copy()

    if finger_count == 0:
        return  # Stop 
    elif finger_count == 1:
        new_pos[0] = max(0, marker_pos[0] - 1)  # up
    elif finger_count == 2:
        new_pos[0] = min(grid_size - 1, marker_pos[0] + 1)  # down
    elif finger_count == 3:
        new_pos[1] = max(0, marker_pos[1] - 1)  # left
    elif finger_count == 4:
        new_pos[1] = min(grid_size - 1, marker_pos[1] + 1)  # right

    # Check for obstacles
    if tuple(new_pos) in obstacles:
        print("Obstacle detected! Possible directions:")
        possible_directions = []
        if marker_pos[0] > 0 and (marker_pos[0] - 1, marker_pos[1]) not in obstacles:
            possible_directions.append("up")
        if marker_pos[0] < grid_size - 1 and (marker_pos[0] + 1, marker_pos[1]) not in obstacles:
            possible_directions.append("down")
        if marker_pos[1] > 0 and (marker_pos[0], marker_pos[1] - 1) not in obstacles:
            possible_directions.append("left")
        if marker_pos[1] < grid_size - 1 and (marker_pos[0], marker_pos[1] + 1) not in obstacles:
            possible_directions.append("right")
        print(", ".join(possible_directions))
    else:
        marker_pos = new_pos
        # Record the new position in the path
        marker_path.append(tuple(marker_pos))

# Function to draw the grid and marker
def draw_grid():
    grid[:] = (0, 0, 0)  # Clear grid

    # Draw obstacles
    for obs in obstacles:
        grid[obs[0], obs[1]] = (0, 0, 255)  # Draw obstacle (red)

    # Draw the marker path
    for i in range(1, len(marker_path)):
        start_pos = marker_path[i - 1]
        end_pos = marker_path[i]
        cv2.line(grid, (start_pos[1], start_pos[0]), (end_pos[1], end_pos[0]), (255, 255, 255), 1)

    # Draw the current marker position
    grid[marker_pos[0], marker_pos[1]] = (0, 255, 0)  # Draw marker (green)

    # Scale up the grid for better visualization
    scaled_grid = cv2.resize(grid, (500, 500), interpolation=cv2.INTER_NEAREST)
    return scaled_grid

# Function to process each frame
def process_frame(frame):
    global last_detected_gesture

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(rgb_frame)

    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Count the number of extended fingers
            fingers_count = count_fingers(hand_landmarks)
            last_detected_gesture = fingers_count

            # Update marker position only if not 5 fingers
            if fingers_count != 5:
                update_marker_position(fingers_count)

            # Display the number of extended fingers on the frame
            cv2.putText(frame, f'Fingers: {fingers_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        # If no hand is detected, continue moving based on the last detected gesture
        if last_detected_gesture is not None and last_detected_gesture != 5:
            update_marker_position(last_detected_gesture)

    return frame

# Capture video stream from webcam
cap = cv2.VideoCapture(0)

# Main loop to process video stream
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get the current time
    current_time = time.time()

    # Check if the update interval has passed
    if current_time - last_update_time >= update_interval:
        # Process frame
        processed_frame = process_frame(frame)

        # Update the last update time
        last_update_time = current_time

    # Draw the grid with the marker
    grid_image = draw_grid()

    # Resize the processed frame to match the height of the grid image
    frame_height, frame_width = processed_frame.shape[:2]
    grid_height, grid_width = grid_image.shape[:2]
    if frame_height != grid_height:
        processed_frame = cv2.resize(processed_frame, (int(frame_width * (grid_height / frame_height)), grid_height))

    # Concatenate the frame and grid image side by side
    combined_image = np.hstack((processed_frame, grid_image))

    # Display the combined image
    cv2.imshow('Hand Gestures Detection and Grid Movement', combined_image)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break        
        
# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Release MediaPipe Hands resources
hands.close()
