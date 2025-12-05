# Author: Endri Dibra 
# Group Project: Explainable Social Navigation
# Task: Speech-to-Text and Text-to-Speech

# Importing the required libraries 
import pyttsx3
import datetime
import speech_recognition as sr


# Initializing recognizer for capturing voice input
recognizer = sr.Recognizer()

# Initializing TTS engine
engine = pyttsx3.init()


# Function for speaking output
def speak(text):

    # Printing assistant message to console
    print("Assistant:", text)

    # Converting text to speech
    engine.say(text)

    # Playing the spoken audio
    engine.runAndWait()


# Function for capturing voice input
def captureVoiceInput():

    # Opening microphone for input
    with sr.Microphone() as source:

        # Adjusting for background noise
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        # Listening for spoken command
        print("Listening for command...")
        audio = recognizer.listen(source)

    # Returning captured audio
    return audio


# Function for converting audio to text
def convertVoicetoText(audio):

    try:
        
        # Converting speech to text using Google API
        text = recognizer.recognize_google(audio)
        
        print("You said:", text)

    except sr.UnknownValueError:

        # Handling case where speech is not recognized
        text = ""
        
        print("Sorry, I didn't understand that.")
        speak("Sorry, I didn't understand that.")

    except sr.RequestError as e:

        # Handling API errors
        text = ""
        
        print("Speech recognition error:", e)
        speak("There was a speech recognition error.")

    # Returning processed text in lowercase
    return text.lower()


# Function for interpreting and processing voice commands
def processVoiceCommand(text):

    # If user says hello
    if "hello" in text:
        
        speak("Hello! Hope you are doing well! How can I help you?")

    # If user asks for the time
    elif "time" in text:
        
        currentTime = datetime.datetime.now().strftime("%H:%M")
        
        speak(f"The current time is {currentTime}.")

    # If user asks for the assistant's name
    elif "name" in text:
        
        speak("Thank you for asking! My name is Minerva, powered by Python.")

    # If user wants to exit the program
    elif "goodbye" in text or "exit" in text:
        
        speak("Exiting program. Goodbye!")
        
        return True

    # If command is not recognized
    else:
        
        speak("Command not recognized. Please try again.")

    # Program continues running
    return False


# Main loop
def main():

    # Boolean controlling program termination
    endProgram = False

    # Running program until user says exit
    while not endProgram:

        # Capturing spoken audio
        audio = captureVoiceInput()

        # Converting audio to text
        text = convertVoicetoText(audio)

        # Only process if speech was recognized
        if text:
        
            endProgram = processVoiceCommand(text)


# Running the main function when the script starts
if __name__ == "__main__":
    
    main()