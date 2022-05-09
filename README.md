# mediapipe

Examples of primarily Mediapipe techniques with help from Opencv. 

FaceMesh provides adequate front-facing close-shots, but loses connection when body is faced away from camera. HolisticPose provide nearby Landmarks of FaceMesh and can detect the direction of the nose/eyes when facing away from camera. Overall, Mediapipe is fast, accurate, but is limited to only one person detected at a time - two or more people in the frame will otherwise confuse each other due to confidence level detections.

MoveNet is another area explored where body landmarks are similar to Mediapipe's Pose Landarks (holistic). MoveNet appears to be slower than Mediapipe, but can detect multiple people's pose.

In both frameworks, object detection was also implemented via yolov5. The goal is to detect frame of vision for each person and identify the object detection bounding boxes. Thus, for each person in the frame, create a list of objects detected (i.e. cars, bus, person, cup, ball, etc. -- based on COCO dataset). 

Further researh into solvePnP method to project a person's frame of vision area onto a 2D plane from X perspective. 

Other idea is to train a custom detection model for person's position in relation to World coordinates so that each 17 points are mapped to the highest probability of direction. 
