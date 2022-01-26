#!/usr/bin/env python
# coding: utf-8

# In[24]:


import cv2
import mediapipe as mp
import time


# In[56]:


class HandDetector():
    def __init__(self,static_image_mode=False,max_hands = 2,model_complexity=0,min_detection_confidence=0.5,
                 min_tracking_confidence=0.6):
        self.static_image_mode = static_image_mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.static_image_mode, self.max_hands,self.model_complexity,
                                        self.min_detection_confidence, self.min_tracking_confidence)        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def findHands(self, img, draw=True):
        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(img, hand_landmarks,self.mp_hands.HAND_CONNECTIONS,
                                                  self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                                   self.mp_drawing_styles.get_default_hand_connections_style())
        return img
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList
    
    def findPositions(self, img,hands = 'all', draw=True):
        hands_positions = {}
        h, w, c = img.shape
        count = 1
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                hand_position = {
                    'wrist':[],
                    'thumb':[],
                    'index':[],
                    'middle':[],
                    'ring':[],
                    'pinky':[]
                }
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id in range(1,5):
                        hand_position['thumb'].append((cx,cy))
                    if id in range(5,9):
                        hand_position['index'].append((cx,cy))
                    if id in range(9,13):
                        hand_position['middle'].append((cx,cy))
                    if id in range(13,16):
                        hand_position['ring'].append((cx,cy))
                    if id in range(16,20):
                        hand_position['pinky'].append((cx,cy))
                    else:
                        hand_position['wrist']=(cx,cy)
                hands_positions[f"hand{count}"] = hand_position
                count+=1
        return hands_positions
    


# In[ ]:




