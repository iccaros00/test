import cv2
import os
import mediapipe as mp

filePath = 'D:/Meadia pipe/Image/6.mkv'

# For webcam input:
if os.path.isfile(filePath):	# 해당 파일이 있는지 확인
  cap = cv2.VideoCapture(filePath)
else:
  print("파일이 존재하지 않습니다.")

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("length :", length)
print("width :", width)
print("height :", height)
print("fps :", fps)

# Tool Select
# 1: face_detection
# 2: Face Mesh
# 3: Selfie Segmentation
# 4: 
Mode = 4

if Mode == 1:      
  mp_face_detection = mp.solutions.face_detection
  mp_drawing = mp.solutions.drawing_utils
  
  with mp_face_detection.FaceDetection(
      model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = face_detection.process(image)

      # Draw the face detection annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.detections:
        for detection in results.detections:
          mp_drawing.draw_detection(image, detection)
          
      cv2.startWindowThread()
      cv2.namedWindow("MediaPipe Face Detection")
      
      # Flip the image horizontally for a selfie-view display.
      # cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
      cv2.imshow('MediaPipe Face Detection', image)
      
      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()
elif Mode == 2:
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_face_mesh = mp.solutions.face_mesh
  drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
  
  with mp_face_mesh.FaceMesh(
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = face_mesh.process(image)

      # Draw the face mesh annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
          
          print(face_landmarks)
          
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
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_contours_style())
          mp_drawing.draw_landmarks(
              image=image,
              landmark_list=face_landmarks,
              connections=mp_face_mesh.FACEMESH_IRISES,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp_drawing_styles
              .get_default_face_mesh_iris_connections_style())
      
      print(mp_face_mesh.FACEMESH_IRISES)
      
      cv2.startWindowThread()
      cv2.namedWindow("MediaPipe Face Mesh")
      
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Face Mesh', image)
      
      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()
elif Mode == 3:
  BG_COLOR = (192, 192, 192) # gray
  
  mp_drawing = mp.solutions.drawing_utils
  mp_selfie_segmentation = mp.solutions.selfie_segmentation

  with mp_selfie_segmentation.SelfieSegmentation(
      model_selection=1) as selfie_segmentation:
    bg_image = None
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # Flip the image horizontally for a later selfie-view display, and convert
      # the BGR image to RGB.
      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      results = selfie_segmentation.process(image)

      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      # Draw selfie segmentation on the background image.
      # To improve segmentation around boundaries, consider applying a joint
      # bilateral filter to "results.segmentation_mask" with "image".
      condition = np.stack(
        (results.segmentation_mask,) * 3, axis=-1) > 0.1
      # The background can be customized.
      #   a) Load an image (with the same width and height of the input image) to
      #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
      #   b) Blur the input image by applying image filtering, e.g.,
      #      bg_image = cv2.GaussianBlur(image,(55,55),0)
      if bg_image is None:
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
      output_image = np.where(condition, image, bg_image)

      cv2.imshow('MediaPipe Selfie Segmentation', output_image)
      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()
elif Mode==4:
  mp_drawing = mp.solutions.drawing_utils
  mp_objectron = mp.solutions.objectron
  
  with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.99,
                            model_name='Shoe') as objectron:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = objectron.process(image)

      # Draw the box landmarks on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.detected_objects:
          for detected_object in results.detected_objects:
              mp_drawing.draw_landmarks(
                image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
              mp_drawing.draw_axis(image, detected_object.rotation,
                                  detected_object.translation)
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Objectron', image)
      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()