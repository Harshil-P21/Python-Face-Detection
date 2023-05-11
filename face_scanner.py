import face_recognition
import os, sys
import cv2
import numpy as np
import math

def face_accuracy(faceDistance, faceMatchValue=0.6):
    range = (1.0 - faceMatchValue)
    linearValue = (1.0 - faceDistance) / (range * 2.0)

    if (faceDistance > faceMatchValue):
        return (str(round(linearValue * 100, 2)) + '%')
    else:
        value = (linearValue + ((1.0 - linearValue) * math.pow((linearValue - 0.5) * 2,0.2))) * 100
        return (str(round(value, 2)) + '%')
    
class FacialRecognition:
    faceLocation = []
    faceEncodes = []
    faceNames = []
    knownEncodes = []
    knownFaces = []
    processCurrentFrame = True

    def __init__(self):
        pass
        self.encode_faces()

    def encode_faces(self):
        # pasrsing through the images folder
        for image in os.listdir('faces'):
            # adding faces to known faces
            faceImage = face_recognition.load_image_file(f'faces/{image}')
            faceEncoding = face_recognition.face_encodings(faceImage)[0]

            self.knownEncodes.append(faceEncoding)
            self.knownFaces.append(image)
        print (self.knownFaces)

    def run_facial_recognition(self):
        # 0 because I only have 1 webcam
        videoCapture = cv2.VideoCapture(0)

        if not videoCapture.isOpened():
            sys.exit(f'Webcam not found')
        
        while True:
            returnOfFrames, frame = videoCapture.read()

            if self.processCurrentFrame:
                # resizing frame to save processing power
                smallFrame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # changing it into RGB using sugar syntax
                rbgSmallFrame = smallFrame[:, :, ::-1]

                # find faces in frame
                self.faceLocation = face_recognition.face_locations(rbgSmallFrame)
                self.faceEncodes = face_recognition.face_encodings(rbgSmallFrame, self.faceLocation)

                # perform image recognition
                self.faceNames = []
                for faceEncoding in self.faceEncodes:
                    # checking for matches with known encodings
                    matches = face_recognition.compare_faces(self.knownEncodes, faceEncoding)

                    # default values (cl is coinfidence level)
                    name = 'Unknown'
                    cl = 'Unknown'

                    faceDistances = face_recognition.face_distance(self.knownEncodes, faceEncoding)

                    bestMatchIndex = np.argmin(faceDistances)
                    

                    if matches[bestMatchIndex]:
                        name = self.knownFaces[bestMatchIndex]
                        cl = face_accuracy(faceDistances[bestMatchIndex])
                    
                    self.faceNames.append(f'{name} ({cl})')

            #do this to only process every other frame
            self.processCurrentFrame = not self.processCurrentFrame

            # display the information
            for (top, right, bottom, left), name in zip(self.faceLocation, self.faceNames):
                # resizing images back to original size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # rectable around face
                cv2.rectangle(frame, (left, top), (right, bottom), (255,0,0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255,0,0), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) == ord('x'):
                break
        videoCapture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    fr = FacialRecognition()
    fr.run_facial_recognition()
    