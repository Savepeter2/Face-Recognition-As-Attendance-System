#!/usr/bin/env python
# coding: utf-8

# In[1]:


import face_recognition
import cv2
import numpy as np
import os
import glob
from datetime import datetime


# In[ ]:





# In[2]:


def markAttendance(name):
    with open(r'C:\Users\USER\Documents\Face Recognition\attendance.csv','r+') as f:
        myDataList = f.readlines()


        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')


# In[ ]:





# In[3]:


faces_encodings = []
faces_names = []
cur_direc = os.getcwd()
path = os.path.join(cur_direc, 'Face Recognition\Train Images')
list_of_files = [f for f in glob.glob(path+'\*')]
number_files = len(list_of_files)
names = list_of_files.copy()


# In[4]:


cur_direc


# In[ ]:





# In[5]:


names


# In[6]:


for i in range(number_files):
    globals()['image_{}'.format(i)] = face_recognition.load_image_file(list_of_files[i])
    globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
    faces_encodings.append(globals()['image_encoding_{}'.format(i)])
# Create array of known names
    names[i] = names[i].replace("C:\\Users\\USER\\Documents\\Face Recognition\\Train Images\\", "")  
    faces_names.append(names[i])


# In[7]:


names


# In[8]:


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


# In[ ]:


video_capture = cv2.VideoCapture(0)
while True:
    #Grab a single frame of video
    ret, frame = video_capture.read()
    
    #Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations( rgb_small_frame)
        face_encodings = face_recognition.face_encodings( rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(faces_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance( faces_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = faces_names[best_match_index]
            face_names.append(name)
    
    
    process_this_frame = not process_this_frame
    
    
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
    # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    # Input text label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        markAttendance(name)
        #markatte
    # Display the resulting image
    cv2.imshow('Video', frame)
# Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# In[ ]:




