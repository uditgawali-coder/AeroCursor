import cv2
import mediapipe as mp
import pyautogui
import math

# SETTINGS 
pyautogui.FAILSAFE = False
SMOOTHING_ALPHA = 0.18
SENSITIVITY = 2.0
NON_LINEAR_POWER = 2.2
DEAD_ZONE = 6
MAX_JUMP = 60

PINCH_DISTANCE = 25
SCROLL_PINCH_DISTANCE = 22
SCROLL_SPEED = 6

VOLUME_SENSITIVITY = 12

screen_w, screen_h = pyautogui.size()

#  MEDIAPIPE 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# STATE 

prev_x, prev_y = pyautogui.position()
smooth_x, smooth_y = prev_x, prev_y

dragging = False
scrolling = False
prev_scroll_y = None

prev_left_y = None

#  FUNCTINS 

def non_linear_gain(dx, dy):
    dist = math.hypot(dx, dy)
    if dist < DEAD_ZONE:
        return 0, 0

    scale = (dist ** NON_LINEAR_POWER) / (dist + 1e-6)
    scale = min(scale, 3.0)

    dx *= scale * SENSITIVITY
    dy *= scale * SENSITIVITY

    dx = max(min(dx, MAX_JUMP), -MAX_JUMP)
    dy = max(min(dy, MAX_JUMP), -MAX_JUMP)

    return dx, dy

# MAIN LOOP
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture_text = "No Gesture"

    if results.multi_hand_landmarks and results.multi_handedness:

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            label = handedness.classification[0].label
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            ix = int(hand_landmarks.landmark[8].x * w)
            iy = int(hand_landmarks.landmark[8].y * h)

            mx = int(hand_landmarks.landmark[12].x * w)
            my = int(hand_landmarks.landmark[12].y * h)

            tx = int(hand_landmarks.landmark[4].x * w)
            ty = int(hand_landmarks.landmark[4].y * h)

#  RIGHT HAND CONTROLS 
            if label == "Right":

                target_x = hand_landmarks.landmark[8].x * screen_w
                target_y = hand_landmarks.landmark[8].y * screen_h

                dx = target_x - prev_x
                dy = target_y - prev_y
                dx, dy = non_linear_gain(dx, dy)

                target_x = prev_x + dx
                target_y = prev_y + dy

                smooth_x += (target_x - smooth_x) * SMOOTHING_ALPHA
                smooth_y += (target_y - smooth_y) * SMOOTHING_ALPHA

                smooth_x = max(0, min(screen_w - 1, smooth_x))
                smooth_y = max(0, min(screen_h - 1, smooth_y))

                pyautogui.moveTo(int(smooth_x), int(smooth_y))
                prev_x, prev_y = smooth_x, smooth_y

                thumb_index_dist = math.hypot(tx - ix, ty - iy)
                index_middle_dist = math.hypot(ix - mx, iy - my)

                #  DRAG 
                if thumb_index_dist < PINCH_DISTANCE:
                    gesture_text = "DRAG"
                    if not dragging:
                        pyautogui.mouseDown()
                        dragging = True
                else:
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False

                #  SCROLL 
                if index_middle_dist < SCROLL_PINCH_DISTANCE and thumb_index_dist > PINCH_DISTANCE:
                    gesture_text = "SCROLL"
                    if not scrolling:
                        scrolling = True
                        prev_scroll_y = iy
                    else:
                        dy_scroll = iy - prev_scroll_y
                        pyautogui.scroll(int(-dy_scroll * SCROLL_SPEED))
                        prev_scroll_y = iy
                else:
                    scrolling = False
                    prev_scroll_y = None

                cv2.circle(frame, (ix, iy), 8, (0,255,0), -1)

            # volume change 
            elif label == "Left":

                if prev_left_y is None:
                    prev_left_y = iy

                dy_vol = iy - prev_left_y

                if abs(dy_vol) > VOLUME_SENSITIVITY:
                    if dy_vol < 0:
                        pyautogui.press("volumeup")
                        gesture_text = "VOLUME UP"
                    else:
                        pyautogui.press("volumedown")
                        gesture_text = "VOLUME DOWN"

                    prev_left_y = iy

                cv2.circle(frame, (ix, iy), 8, (255,0,255), -1)

    # Camera tab

    cv2.rectangle(frame, (10,10), (320,60), (0,0,0), -1)
    cv2.putText(frame, f"Gesture: {gesture_text}", (20,45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.imshow("PRO AI Hand Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()