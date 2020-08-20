from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing import image
#from keras.callbacks import TensorBoard
def Prepair_Image(test_image):
    test_image = cv2.resize(test_image, (128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    return test_image
def Camera_Onn():
    model = load_model('C:/Users/MohsinQuddus/PycharmProjects/FaceMaskDetection/Model.h5')
    face_cascade = cv2.CascadeClassifier('C:/Users/MohsinQuddus/PycharmProjects/FaceMaskDetection/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray=frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, scaleFactor=1.32, minNeighbors=5,minSize=(30, 30))
        for (x, y, w, h) in face:
            test_image = gray[y:y + h, x:x + w]
            test_image=Prepair_Image(test_image)
            prediction = model.predict(test_image)
            if int(prediction[0][0]) == 1:
                print(prediction[0][0])
                name = "NOMask"
            elif int(prediction[0][0]) == 0:
                name="Mask"
            #name On face
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = name
            color = (0, 255, 0)
            stroke = 1
            cv2.putText(frame, name, (x, y), font, 1, color,stroke,cv2.LINE_AA)
            # rectangle on face
            color = (0, 255, 0)
            stroke = 2
            width = x + w
            height = y + h
            cv2.rectangle(frame, (x, y), (width, height), color, stroke)
        cv2.imshow("Frame",frame)
        if cv2.waitKey(20) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    Camera_Onn()
    cv2.destroyAllWindows()