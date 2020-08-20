import cv2
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
cv2.namedWindow("Window")
count=0
while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray , 1.3 , 5)
    for (x,y,w,h) in faces:
        roi_gray=gray[y:y+h,x:x+w]
        roi_color = frame[y:y + h, x:x + w]
        imgpath='Dataset/training_set/WithoutMask/pic'+str(count)+'.jpg'
        count+=1
        cv2.imwrite(imgpath, roi_gray)
        color=(0,255,0)
        stroke=2
        width=x+w
        height=y+h
        cv2.rectangle(frame,(x,y),(width,height),color,stroke)

    cv2.imshow("Window", frame)
    video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    #This breaks on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if count==700:
        break
#--------------------------
video_capture.release()
cv2.destroyAllWindows()
print("Samples Collected.")