import mediapipe as mp
import cv2 as cv

webcam=cv.VideoCapture(0)
mp_draw=mp.solutions.drawing_utils
mp_holistic=mp.solutions.holistic
##holistic mediapipe kütüphanesi. tek bir kütüphane
#vucut,el gibi şeyleri tespit edebiliyoruz. 


with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
#burada minimum hareket oranı vs. verdik.
    while True:
        isTrue,frame=webcam.read()
       # rgb=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        #bgr=cv.cvtColor(rgb,cv.COLOR_RGB2BGR)
        
        frame=cv.resize(frame,(1024,720))
        result=holistic.process(frame)
        #burada sol ve sag eli tanıtma yapıcaz
        if result.left_hand_landmarks:
            mp_draw.draw_landmarks(frame,result.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
            
        if result.right_hand_landmarks:
            mp_draw.draw_landmarks(frame,result.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
            
    
        if cv.waitKey(20) & 0xFF==ord('q'):
            break
        
    
        cv.imshow('kamera',frame)  
  
webcam.release()
cv.destroyAllWindows()    
