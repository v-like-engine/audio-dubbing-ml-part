import face_recognition
import numpy as np
import cv2
import os
import time
# imports and constant preferences


CHARACTERS_PATH = '../../actors'


def extract_actors() -> list:
    """
    Function to extract face details for all of the characters in the given path (CHARACTERS_PATH constant used)
    :return: tuple of two lists: list of character name, list of face details, one sublist for each character
    """
    face_images = []
    face_character_names = []
    character_face_encodings = []
    for chc in os.listdir(CHARACTERS_PATH):
        curIm = cv2.imread(f'{CHARACTERS_PATH}/{chc}')
        face_images.append(curIm)
        face_character_names.append(os.path.splitext(chc)[0])
    for img in face_images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        character_face_encodings.append(encode)
    return face_character_names, character_face_encodings


char_names, char_encodings = extract_actors()
stream = 'rtmp://rtmp.example.com/live/test'
video = '../../sample_videos/kianu.mp4'

capture = cv2.VideoCapture(video)
cv2.cvtColor(capture, cv2.COLOR_BGR2RGB)
if (capture.isOpened() == False):
    print("Error opening the video file")
# Read fps and frame count
else:
    # Get frame rate information
    fps = capture.get(cv2.CAP_PROP_FPS)
    print('Frames per second : ', fps,'FPS')

    # Get frame count
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Frame count : ', frame_count)

    i = 0
 
while(capture.isOpened()):
    i += 1
    # capture.read() methods returns a tuple, first element is a bool 
    # and the second is frame
    if i > 99:
        break
    ret, frame = capture.read()
    print(ret, i)
    try:
        imgSmall = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
        imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(imgSmall)
        current_frame_encodings = face_recognition.face_encodings(imgSmall, face_locations)
        for encoding, loc in zip(current_frame_encodings, face_locations):
            matches = face_recognition.compare_faces(char_encodings, encoding)
            distance_faces = face_recognition.face_distance(char_encodings, encoding)
            match_idx = np.argmin(distance_faces)
            if matches[match_idx]:
                name = char_names[match_idx].upper()
                print(name)
            print(distance_faces)
    except Exception:
        break

 
# Release the video capture object
capture.release()
cv2.destroyAllWindows()
