import cv2
from deepface import DeepFace

# Path to the human image
img_path = r'C:\Users\Blaise Fonguh\OneDrive\Documents\GitHub\face recognition\me.jpg'

# Load the human image for comparison
img = cv2.imread(img_path)

def identify_human(frame):
    try:
        detected_faces = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)
        
        for face in detected_faces:
            result = DeepFace.verify(face, img, model_name='Facenet')
            if result['verified']:
                return True
    except Exception as e:
        print("Error:", e)
    
    return False

# Path to the video file
video_path = r'C:\Users\Blaise Fonguh\OneDrive\Documents\GitHub\face recognition\me_2.mp4'

# Open the video capture
video = cv2.VideoCapture(video_path)

while video.isOpened():
    ret, frame = video.read()

    if ret:
        if identify_human(frame):
            cv2.putText(frame, "Identified", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Not Identified", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Identifying Human in Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()
