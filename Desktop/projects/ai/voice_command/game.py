import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
import cv2
import turtle
import time  # Import time for the delay

# Load your trained model
model = tf.keras.models.load_model('model4.h5')

# Command labels corresponding to your model output
command_labels = ['down', 'go', 'left', 'right', 'up']

# Function to record audio with a specified buffer duration
def record_audio(duration=1, sample_rate=16000):
    print("Get ready to speak your command...")
    time.sleep(1)  # Wait for 1 second to allow the user to prepare
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("Recording finished.")
    return audio.flatten()

# Function to preprocess the audio for the model
def preprocess_audio(audio, sr=16000):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    resized_mfcc = cv2.resize(mfcc, (128, 128))
    resized_mfcc = resized_mfcc[..., np.newaxis]  # Add channel dimension
    return resized_mfcc

# Function to classify the audio command
def classify_command(audio):
    mfcc = preprocess_audio(audio)
    mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
    prediction = model.predict(mfcc)
    predicted_class = np.argmax(prediction, axis=1)[0]
    if predicted_class not in [0,1,2,3,4]:
        return 'silence'
    return command_labels[predicted_class]

# Function to execute the turtle movement based on the command
def execute_command(command):
    if command == "up":
        t.setheading(90)  # Point the turtle up
    elif command == "down":
        t.setheading(270)  # Point the turtle down
    elif command == "left":
        t.setheading(180)  # Point the turtle left
    elif command == "right":
        t.setheading(0)    # Point the turtle right
    elif command == "go":
        t.forward(50)      # Move forward

# Function to record, classify, and execute command when "r" is pressed
def on_key_press():
    audio = record_audio(duration=2)  # Record audio for 2 seconds
    command = classify_command(audio)  # Classify the command
    print(f"Command recognized: {command}")
    execute_command(command)

# Set up the turtle screen and key press event
screen = turtle.Screen()
screen.title("Sajed's voice command turtle game")
t = turtle.Turtle()

# Set up the turtle
t.speed(1)  # Set the turtle speed to slow for better visibility
t.shape("turtle")  # Optional: Change the turtle shape

# Bind the "r" key to the command recording and execution function
screen.listen()
screen.onkey(on_key_press, "r")

# Keep the window open until closed by the user
turtle.done()
