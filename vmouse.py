#!/usr/bin/env python
# coding: utf-8



# In[39]:


import cv2
import numpy as np
import sys
import mouse
#sys.path.insert(1, 'F:/machine learning/modules/')
import handLandmarkMP as hlm
import pyautogui
import time


# In[11]:


screen_width, screen_height= pyautogui.size()


# In[12]:


print("width: {}, height: {}".format(screen_width,screen_height))


# In[13]:


screen_height//3


# In[14]:


hands = hlm.HandDetector()


# In[15]:


img_center = (img.shape[1]//2,img.shape[0]//2)
x1,y1 = img_center[0] - (455//2), img_center[1] - (256//2)
x2,y2 = img_center[0] + (455//2), img_center[1] + (256//2)
venv_rec_x = x2 - x1
venv_rec_y = y2 - y1
venv_rec_y


# In[29]:


np.isclose([11, 9], [0, 0],atol = 10)


# In[81]:


cap = cv2.VideoCapture(0)
success, img = cap.read()
img_center = (img.shape[1]//2,img.shape[0]//2)
x1,y1 = img_center[0] - (455//2), img_center[1] - (256//2)
x2,y2 = img_center[0] + (455//2), img_center[1] + (256//2)
a,b,c,d = x1,y1,x2,y2
e,f,g,h = 1,1,1366,768

start = time.time()

avg_factor = 7
avg_index = 0
history_x = np.ones(avg_factor) * 1366
history_y = np.ones(avg_factor) * 768
x_before = 1366 // 2
y_before = 768 // 2

start_point = (x1,y1)
end_point = (x2,y2)
rectangle_color = (155,155,155)
rectangle_thickness = 3
frames = 0

font = cv2.FONT_HERSHEY_SIMPLEX

while cap.isOpened():
    success, img = cap.read()
    #img = cv2.resize(img,(screen_width//2,screen_height//2))
    img = cv2.flip(img, 1)
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
    img = hands.findHands(img)
    hands_positions = hands.findPositions(img)
    #print(hands_positions)
    #mouse.move(screen_width//2, screen_height//2, absolute=False, duration=0)
    img = cv2.rectangle(img, start_point, end_point, rectangle_color, rectangle_thickness)
    if hands_positions:
        index = np.array(hands_positions['hand1']['index'][-1])
        middle = np.array(hands_positions['hand1']['middle'][-1])
        ring = np.array(hands_positions['hand1']['ring'][-1])
        #print(f"middle co-ordinates : {middle}")
        #print(f"index co-ordinates : {index}")
        
        # get the ecludian distance between middle finger tip and index finger tip
        x, y = np.abs(middle-index)
        dist = int((x**2 + y**2)**0.5)
        print(dist)
        
        #print(f"index pos:{index} middle pos {middle}")
        
        # map x and y values from the frame to the screen size
        x_ = e + (middle[0] - a) * (g - e) / (c - a)
        y_ = f + (middle[1] - b) * (h - f) / (d - b)
        
        if (np.isclose(dist,60,atol = 7)):
            mouse.click(button='left')
        
        if (np.all(np.isclose([x_,y_],[x_before,y_before],atol = 10))):
            x_ = x_before
            y_ = y_before
        
        # appending current average
        history_x[avg_index] = int(x_)
        history_y[avg_index] = int(y_)
        
        # averaging over position history
        pos_avg_x = int(np.average(history_x))
        pos_avg_y = int(np.average(history_y))
        
        #print(history_x)
        
        mouse.move(pos_avg_x, pos_avg_y, absolute=True, duration=0)
        #cv2.circle(img, hands_positions['hand1']['middle'][-1], 0, (0, 0, 0), 6)
    
    x_before = x_
    y_before = y_
    end = time.time()
    avg_index += 1
    
    if (avg_index / avg_factor == 1):
        avg_index = 0
    fps = str(int(frames / (end-start)))
    frames += 1
    cv2.putText(img, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow("Image", img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[21]:


mouse.move(1366, 640, absolute=True, duration=0.1)


# In[ ]:




