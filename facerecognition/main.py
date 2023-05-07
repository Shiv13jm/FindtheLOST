import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
 
video_capture = cv2.VideoCapture(0)
 
Sundar_Pichai_image = face_recognition.load_image_file("photos\sundarpichai.jpeg")
Sundar_Pichai_encoding = face_recognition.face_encodings(Sundar_Pichai_image)[0]

Taylor_Swift_image = face_recognition.load_image_file("photos\Taylor Swift at The Eras Tour #taylorswift #theerastour.jpeg")
Taylor_Swift_encoding = face_recognition.face_encodings(Taylor_Swift_image)[0]
 
barack_image = face_recognition.load_image_file("photos/BARACK OBAMA GLOSSY POSTER PICTURE PHOTO president official white house usa 1538  _ eBay.jpeg")
barack_encoding = face_recognition.face_encodings(barack_image)[0]
 
messi_image = face_recognition.load_image_file("photos/The Sun_ Messi wants Salah in Barcelona _ Sada Elbalad.jpeg")
messi_encoding = face_recognition.face_encodings(messi_image)[0]
 
known_face_encoding = [

Taylor_Swift_encoding,
barack_encoding,
messi_encoding
]
 
known_faces_names = [
"Sundar_Pichai",
"Taylor Swift",
"Barack Obama",
"Messi"
]
 
people = known_faces_names.copy()
 
face_locations = []
face_encodings = []
face_names = []
s=True
 
 
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
 
 
 
f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)
 
while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
 
            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale              = 1.5
                fontColor              = (255,0,0)
                thickness              = 3
                lineType               = 2
 
                cv2.putText(frame,name+' found ', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
 
                if name in people:
                    people.remove(name)
                    print(people)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
video_capture.release()
cv2.destroyAllWindows()
f.close()