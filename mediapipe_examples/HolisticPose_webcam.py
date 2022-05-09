import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(color=(0,255,0),thickness=1, circle_radius=1)
mp.solutions.pose.PoseLandmark
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture('input/dance.mp4')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output/HolisticPose2.avi', fourcc, 30.0, (640,480))

while cap.isOpened():
    success, image = cap.read()

    image = cv2.cvtColor(cv2.flip(image, 2), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)

    if results.pose_landmarks:
        # Draw landmark annotation on the image.
        print(results.pose_landmarks)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_h, img_w, img_c = image.shape            
            



        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)  

        # Draw pose, left and right hands, and face landmarks on the image.
        mp_drawing.draw_landmarks(
                image = image,
                landmark_list = results.face_landmarks,
                connections = mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec = None,
                connection_drawing_spec = mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
                image = image,
                landmark_list = results.pose_landmarks,
                connections = mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec = mp_drawing_styles.
                get_default_pose_landmarks_style())

    
    cv2.imshow('Head Pose Estimation', image)
    out.write(image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
out.release()
cap.release()
cv2.destroyAllWindows()