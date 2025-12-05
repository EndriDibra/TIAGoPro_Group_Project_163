# Author: Endri Dibra 
# Group Project: Explainable Social Navigation
# Task: Face Recognition and Text to Speech

# Importing the required libraries 
import cv2
import time
import queue
import cvzone
import pyttsx3
import threading
import face_recognition
from deepface import DeepFace


# Initializing TTS engine, to speak the given commands
engine = pyttsx3.init()

# Queue for text-to-speech
speechQueue = queue.Queue()


# Function run by background thread
def speechWorker():

    while True:
    
        # Blocks until text available
        text = speechQueue.get()  

        # Sentinel value to stop the thread
        if text is None:          
           
            break
        
        engine.say(text)
        engine.runAndWait()
        speechQueue.task_done()


# Starting background speech thread
threading.Thread(target=speechWorker, daemon=True).start()


# Function for main loop
def speak(text):
    
    speechQueue.put(text)


# Loading authorized face names from file into a set
def loadAuthorizedNames(filename):
    
    try:
       
       # Openning file
        with open(filename, "r") as file:
            
            # Getting authorised names of personnel
            names = {line.strip() for line in file if line.strip()}
       
            print(f"Loaded {len(names)} authorized face names.")
       
            return names
    
    except FileNotFoundError:
    
        print(f"Authorized faces file '{filename}' not found. Starting with empty set.")
    
        return set()


# loading people face encodings and names from image files
def facesLoading(image_paths, names):

    # lists to store face encodings, names, images of each person
    peopleFaceEncodings = []
    
    peopleNames = []

    peopleImages = []

    # looping through the images and names of people
    for imagePath, name in zip(image_paths, names):
        
        # loading the image of a person
        image = face_recognition.load_image_file(imagePath)

        # using all face encodings found in the image
        faceEncodings = face_recognition.face_encodings(image)
        
        peopleFaceEncodings.extend(faceEncodings)
        
        peopleNames.extend([name] * len(faceEncodings))

        # loading the image for later display
        personImage = cv2.imread(imagePath)

        # for each face encoding, appending the corresponding image
        for _ in range(len(faceEncodings)):
            
            peopleImages.append(personImage)

    return peopleFaceEncodings, peopleNames, peopleImages


# utilizing face recognition in a camera stream
def recognizeFaces(camera, peopleFaceEncodings, peopleNames, peopleImages):

    # loading gender detection models
    genderProto = "weights/gender_deploy.prototxt"
    genderModel = "weights/gender_net.caffemodel"
    
    # loading age detection models
    ageProto = "weights/age_deploy.prototxt"
    ageModel = "weights/age_net.caffemodel"

    # gender detection model
    genderNet = cv2.dnn.readNetFromCaffe(genderProto, genderModel)
    
    # age detection model
    ageNet = cv2.dnn.readNetFromCaffe(ageProto, ageModel)
    
    # the mean values used in the model
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    
    # a list storing person's possible genders
    genderList = ['Male', 'Female']
    
    # a list storing person's possible ages
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(22-25)', '(26-32)', '(38-43)', '(48-53)', '(60-100)']
    
    # To prevent repeating speech/same name too frequently
    lastSpokenName = None

    # Variables for FPS calculation
    prevTime = 0
    fps = 0

    # looping through the camera frames
    while camera.isOpened():

        # capturing each frame from the camera stream
        success, frame = camera.read()
        
        # if camera did not work, program will stop
        if not success:

            print("Error! Camera could not be opened.")
            
            break
        
        # finding all face locations and encodings in the current frame
        faceLocations = face_recognition.face_locations(frame)

        for faceLocation in faceLocations:
            
            # Face location coordinates
            top, right, bottom, left = faceLocation
            
            # extracting the face encoding for the current face
            faceEncoding = face_recognition.face_encodings(frame, [faceLocation])[0]

            # comparing the face with the known faces
            matches = face_recognition.compare_faces(peopleFaceEncodings, faceEncoding, tolerance=0.6)
            
            # this variable will be used in case that a face cannot be recognized
            name = "Unknown Person"

            # placeholder for the corresponding image
            imageToDisplay = None

            # if a match is found, the right name will be displayed
            if True in matches:

                firstMatchIndex = matches.index(True)
                
                name = peopleNames[firstMatchIndex]

                imageToDisplay = peopleImages[firstMatchIndex]
                
            # extracting face ROI for emotion analysis
            faceROI = frame[top:bottom, left:right]

            resizedFaceROI = cv2.resize(faceROI, (224, 224))
            
            # predicting emotion with DeepFace.analyze
            try:

                # resizing the ROI for DeepFace if needed
                resizedFaceROI = cv2.resize(faceROI, (224, 224))
                
                # using the 'Emotion' model or another available emotion-specific model
                emotionAnalysis = DeepFace.analyze(resizedFaceROI, actions=['emotion'], enforce_detection=False)
                
                # checking if the result is in the expected format
                if isinstance(emotionAnalysis, list) and len(emotionAnalysis) > 0:
                
                    emotion = emotionAnalysis[0].get('dominant_emotion', "Unknown")
                
                # in case that an unknown emotion is detected
                else:
                
                    emotion = "Unknown"

            # error handling process
            except Exception as e:
                
                print("DeepFace Emotion Detection Error:", e)
                
                emotion = "Error"

            # predicting gender and age
            blob = cv2.dnn.blobFromImage(faceROI, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            genderNet.setInput(blob)
            
            genderPrediction = genderNet.forward()
            
            # selecting the highest score about gender
            gender = genderList[genderPrediction[0].argmax()]

            ageNet.setInput(blob)
            
            agePrediction = ageNet.forward()
            
            # selecting the highest score about age
            age = ageList[agePrediction[0].argmax()]

            # Defining bounding box width and height
            bboxWidth = right - left
            
            bboxHeight = bottom - top
            
            bbox = (left, top, bboxWidth, bboxHeight)

            # Setting rectangle color: blue if authorized, red if unauthorized
            if name in personNames:
                
                # BGR: Blue rectangle
                colorR = (255, 0, 0) 
                
            else:
                
                # BGR: Red rectangle
                colorR = (0, 0, 255)  

            # drawing rectangle and display information on the frame
            cvzone.cornerRect(frame, bbox, colorR=colorR, colorC=(0, 255, 0))
            
            cv2.putText(frame, f"Name: {name}", (right + 20, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            cv2.putText(frame, f"Gender: {gender}", (right + 20, top + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            cv2.putText(frame, f"Age: {age}", (right + 20, top + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
            
            cv2.putText(frame, f"Emotion: {emotion}", (right + 20, top + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            # Voice Feedback, Text-to-Speech 
            if name != lastSpokenName:
                
                if name == "Unknown Person":
                
                    speak("Hello, you are not an operator!")
                
                else:
                
                    speak(f"Hello, operator {name}!")
                
                lastSpokenName = name

            # image/photo of the current person to be displayed
            if imageToDisplay is not None:
                
                # resizing the person's image/photo
                imageToDisplay = cv2.resize(imageToDisplay, (120, 120))

                # calculating position in the top right corner
                frameHeight, frameWidth = frame.shape[:2]
                
                # offset from right edge
                topRightX = frameWidth - 130

                # offset from top edge
                topRightY = 10
            
                # adding the image on the frame
                frame[topRightY:topRightY + 120, topRightX:topRightX + 120] = imageToDisplay

        # calculating FPS based on total frame processing time
        currTime = time.time()
        fps = 1 / (currTime - prevTime) if (currTime - prevTime) > 0 else 0
        prevTime = currTime

        # overlay FPS on the frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # displaying the resulting frame
        cv2.imshow('Face Recognition', frame)

        # terminating video with letter "q:quit"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            break

    # releasing the camera
    camera.release()

    # closing all the windows
    cv2.destroyAllWindows()

# specifing the paths to people's image files
personImagePaths = [

    "faceimages/Endri.jpg",
    "faceimages/Daniel.png",
    "faceimages/Nicklas.jpeg",
    "faceimages/Joel.jpeg",
    "faceimages/Elon.jpeg",
    "faceimages/Trump.jpg"

]

# specifing the names associated with each person
personNames = [

    "Endri",
    "Daniel",
    "Nicklas",
    "Joel",
    "Elon Musk",
    "Donald Trump"
]

# loading the known face encodings and names
knownFaceEncodings, knownNames,knownImages = facesLoading(personImagePaths, personNames)

# opening the default camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# running face recognition on the live video stream
recognizeFaces(camera, knownFaceEncodings, knownNames, knownImages) 