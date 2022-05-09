import torch
import numpy as np
import cv2
from time import time

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(color=(0,255,0),thickness=1, circle_radius=1)
#mp.solutions.pose.PoseLandmark
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output/yolov5-mediapipe2.avi', fourcc, 30, (640,480))

class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """
    def __init__(self, capture_index):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.capture_index = capture_index
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
      
        return cv2.VideoCapture(self.capture_index)

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = self.get_video_capture()
        assert cap.isOpened()
      
        while True:
            #start_time = time()
            ret, frame = cap.read()
            assert ret
            
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)


            image = frame.copy()
            #image.flags.writeable = False
            results2 = holistic.process(image)

            face_3d = []
            face_2d = []
            img_h, img_w, img_c = image.shape
            if results2.pose_landmarks:
                for idx, lm in enumerate(results2.pose_landmarks.landmark):
                    if idx == 0 or idx == 5 or idx == 2 or idx == 4 or idx == 1 or idx == 3 or idx == 6:
                        if idx == 0: # nose
                            Leye_2d = (lm.x * img_w, lm.y * img_h)
                            Leye_3d = (lm.x * img_w, lm.y * img_h, lm.z * 4000)

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


                # Draw landmark annotation on the image.
                #print(results2.pose_landmarks)
                image.flags.writeable = True
                #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                img_h, img_w, img_c = image.shape            
                print(f'Image shape: {image.shape}')
                fps = cap.get(cv2.CAP_PROP_FPS)
                cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)  

                # Draw pose, left and right hands, and face landmarks on the image.
                mp_drawing.draw_landmarks(
                        image = image,
                        landmark_list = results2.face_landmarks,
                        connections = mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec = None,
                        connection_drawing_spec = mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                        image = image,
                        landmark_list = results2.pose_landmarks,
                        connections = mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec = mp_drawing_styles.
                        get_default_pose_landmarks_style())

                # Display the nose direction
                Leye_3d_projection, jacobian = cv2.projectPoints(Leye_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                p1 = (int(Leye_2d[0]), int(Leye_2d[1]))
                p2 = (int(Leye_2d[0] + y * 10) , int(Leye_2d[1] - x * 10))

                cv2.line(image, p1, p2, (255, 0, 0), 3)
                # Add the text on the image
                cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                
                #end_time = time()
                #t - end_time - start_time
                #fps = 1 / t
                out.write(image)
                #print(f"Frames Per Second : {fps}")                
                #cv2.imshow('YOLOv5 Detection', frame)   # only yolov5 obj detections
                cv2.imshow('YOLOv5 with MediaPipe', image)   # yolov5 obj detection with medipipe
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
      
        cap.release()
        out.release()
        
# Create a new object and execute.
#detector = ObjectDetection(capture_index='input/dance.avi') # File input
detector = ObjectDetection(capture_index=0) # Webcam
detector()


