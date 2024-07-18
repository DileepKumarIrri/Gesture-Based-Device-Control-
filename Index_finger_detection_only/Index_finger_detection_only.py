import cv2
import mediapipe as mp
import numpy as np

cursor_radius = 10
cursor_color = (0, 255, 0)  
click_color = (0, 0, 255)  

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)


cursor_positions = []

def draw_cursor(img, position, clicked=False):
    color = click_color if clicked else cursor_color
    cv2.circle(img, position, cursor_radius, color, -1)

def apply_moving_average(new_position):
    cursor_positions.append(new_position)
    if len(cursor_positions) > 3: 
        cursor_positions.pop(0)
    return np.mean(cursor_positions, axis=0).astype(int)

def main():
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            continue

    
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        img_black = np.zeros(img.shape, dtype=np.uint8)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                thumb_tip_pos = np.array([thumb_tip.x, thumb_tip.y])
                index_finger_tip_pos = np.array([index_finger_tip.x, index_finger_tip.y])

                width, height = img.shape[1], img.shape[0]
                thumb_tip_pos_scaled = np.multiply(thumb_tip_pos, [width, height]).astype(int)
                index_finger_tip_pos_scaled = np.multiply(index_finger_tip_pos, [width, height]).astype(int)
                smoothed_position = apply_moving_average(index_finger_tip_pos_scaled)
                
                
                distance = np.linalg.norm(thumb_tip_pos_scaled - index_finger_tip_pos_scaled)
                clicked = distance < 20

                draw_cursor(img_black, smoothed_position, clicked)

        cv2.imshow("Hand Tracking", img_black)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
