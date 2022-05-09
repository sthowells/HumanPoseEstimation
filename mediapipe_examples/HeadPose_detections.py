import cv2
import mediapipe as mp
import numpy as np
import time
from decimal import Decimal

# Mediapipe - FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))
mp_drawing_styles = mp.solutions.drawing_styles

# Mediapipe - Selfie
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
MASK_COLOR = (192,192,192)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output/EyesDirection_withDetections.avi', fourcc, 30.0, (640,480))

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    bg_image = None
    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    ### Detect boxes ###
    # Fill rectangular contours
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,9)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (255,255,255), -1)

    # Morph open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)

    # Draw rectangles
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)

    ############

    # To improve performance
    image.flags.writeable = True

    #### Seg ###
    seg_image = image.copy()
    seg_image = cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR)  
    seg_image_results = selfie_segmentation.process(seg_image)
    
    condition = np.stack((seg_image_results.segmentation_mask,) * 3, axis=-1) > 0.1
        
    if bg_image is None:
      bg_image = np.zeros(image.shape, dtype=np.uint8)
      bg_image = cv2.GaussianBlur(image,(55,55),0)
      bg_image[:] = MASK_COLOR 

    output_image = np.where(condition, seg_image, bg_image)
    ############

    # Get the result
    results = face_mesh.process(image)
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    # Nose: 1, Left Eye: 33; Right Eye: 263 
                    # Left eye indices list: LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
                    # Right eye indices list: RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
                    if idx == 263:
                        Leye_2d = (lm.x * img_w, lm.y * img_h)
                        Leye_3d = (lm.x * img_w, lm.y * img_h, lm.z * 4000)
                    if idx == 33:
                        Reye_2d = (lm.x * img_w, lm.y * img_h)
                        Reye_3d = (lm.x * img_w, lm.y * img_h, lm.z * 4000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       
            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
          
            # See where the user's head tilting
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Forward"

            # Display the nose direction
            Leye_3d_projection, jacobian = cv2.projectPoints(Leye_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            Reye_3d_projection, jacobian = cv2.projectPoints(Leye_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(Leye_2d[0]), int(Leye_2d[1]))
            p2 = (int(Leye_2d[0] + y * 10) , int(Leye_2d[1] - x * 10))

            p3 = (int(Reye_2d[0]), int(Reye_2d[1]))
            p4 = (int(Reye_2d[0] + y * 10) , int(Reye_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)
            cv2.line(image, p3, p4, (0, 0, 255), 3)
            # Add the text on the image
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                    
        mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

    out.write(image)
    cv2.imshow('Head Pose Estimation', image)
    cv2.imshow('Seg', output_image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()