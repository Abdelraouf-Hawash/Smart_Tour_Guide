# text to speech library
#pyttsx3 is a text-to-speech conversion library in Python. Unlike alternative libraries, it works offline, and is compatible with both Python 2 and 3
import pyttsx3
# library convert audio into text
import speech_recognition as sr
# library to display Web-based documents to users
import webbrowser
# library to fetch current date and time
#We must import the datetime module in order to give our assistant the current time. The command to import this module into your programme is 
import datetime
import calendar
# library to fetch wikipedia information ,Wikipedia:- 
# As we all know Wikipedia is a great source of knowledge just like GeeksforGeeks we have used the Wikipedia module to get information from Wikipedia or to perform a Wikipedia search. To install this module type the below command in the terminal.
import wikipedia
# library to play youtube video and perform google search
import pywhatkit
# library that is used to create one-line jokes
import pyjokes
#library to start play ur sound
from playsound import playsound
#library to translate from google
from googletrans import Translator
#Libaray for Google Text-to-Speech
from gtts import gTTS
#library to portable way of using operating system dependent functionality.
import os
#the de facto standard for making HTTP requests in Python
import requests
import requests, json
'''
import subprocess
import wolframalpha
import tkinter
import json
import random
import operator
import os
import winshell
import feedparser
import smtplib
import ctypes
import time
import requests
import shutil
from twilio.rest import Client
#from clint.textui import progress
from ecapture import ecapture as ec
from bs4 import BeautifulSoup
import win32com.client as wincl
from urllib.request import urlopen
'''
#--------------------------------------------
#For proper communication between the user and the assistance, audio is now required. Therefore, we will set up the pyttsx3 module.

#What is PyTTTX3?


#It is a Python module that will enable us to translate text into speech.
#Additionally, it operates offline and is compatible with both Python 2 and Python 3.

#we will set our engine to Pyttsx3 which is used for text to speech in Python and 
#sapi5 is a Microsoft speech application platform interface we will be using this for text to speech function.
#---------------------------------------------------
#Describe sapi5.

#Microsoft offers voice detection and synthesis technology under the name Microsoft Speech API (SAPI5).
#Usually, it aids with voice synthesis and recognition.
#-----------------------------------------------------
#Describe VoiceId.

#Voice ID enables us to choose from a variety of voices.

#voice[0].id = Voice of a man, voice[1].voice = Female
#-----------------------------------------------------



engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
# change the voice Id to “0” for the Male voice while using assistant here we are using a Female voice for all text to speech
engine.setProperty('voice', voices[1].id)

# dictionary of all the common languages in the world
dic=('afrikaans', 'af', 'albanian', 'sq', 'amharic', 'am', 
     'arabic', 'ar', 'armenian', 'hy', 'azerbaijani', 'az',
 'basque', 'eu', 'belarusian', 'be', 'bengali', 'bn', 'bosnian',
     'bs', 'bulgarian', 'bg', 'catalan', 'ca',
  'cebuano', 'ceb', 'chichewa', 'ny', 'chinese (simplified)',
     'zh-cn', 'chinese (traditional)', 'zh-tw',
  'corsican', 'co', 'croatian', 'hr', 'czech', 'cs', 'danish',
     'da', 'dutch', 'nl', 'english', 'en', 'esperanto',
  'eo', 'estonian', 'et', 'filipino', 'tl', 'finnish', 'fi', 
     'french', 'fr', 'frisian', 'fy', 'galician', 'gl',
  'georgian', 'ka', 'german', 'de', 'greek', 'el', 'gujarati', 
     'gu', 'haitian creole', 'ht', 'hausa', 'ha', 
  'hawaiian', 'haw', 'hebrew', 'he', 'hindi', 'hi', 'hmong', 
     'hmn', 'hungarian', 'hu', 'icelandic', 'is', 'igbo',
  'ig', 'indonesian', 'id', 'irish', 'ga', 'italian', 'it', 
     'japanese', 'ja', 'javanese', 'jw', 'kannada', 'kn',
  'kazakh', 'kk', 'khmer', 'km', 'korean', 'ko', 'kurdish (kurmanji)',
     'ku', 'kyrgyz', 'ky', 'lao', 'lo', 
  'latin', 'la', 'latvian', 'lv', 'lithuanian', 'lt', 'luxembourgish',
     'lb', 'macedonian', 'mk', 'malagasy',
  'mg', 'malay', 'ms', 'malayalam', 'ml', 'maltese', 'mt', 'maori',
     'mi', 'marathi', 'mr', 'mongolian', 'mn',
  'myanmar (burmese)', 'my', 'nepali', 'ne', 'norwegian', 'no',
     'odia', 'or', 'pashto', 'ps', 'persian',
   'fa', 'polish', 'pl', 'portuguese', 'pt', 'punjabi', 'pa',
     'romanian', 'ro', 'russian', 'ru', 'samoan',
   'sm', 'scots gaelic', 'gd', 'serbian', 'sr', 'sesotho', 
     'st', 'shona', 'sn', 'sindhi', 'sd', 'sinhala',
   'si', 'slovak', 'sk', 'slovenian', 'sl', 'somali', 'so', 
     'spanish', 'es', 'sundanese', 'su', 
  'swahili', 'sw', 'swedish', 'sv', 'tajik', 'tg', 'tamil',
     'ta', 'telugu', 'te', 'thai', 'th', 'turkish', 'tr',
  'ukrainian', 'uk', 'urdu', 'ur', 'uyghur', 'ug', 'uzbek', 
     'uz', 'vietnamese', 'vi', 'welsh', 'cy', 'xhosa', 'xh',
  'yiddish', 'yi', 'yoruba', 'yo', 'zulu', 'zu')
#-------------------------------------------------------------------
#Writing our own speak() function: 
def speak(audio):
    engine.say(audio)
    engine.unAndWait()


def username():
    speak("What should i call you ")
    uname = LISTENING()
    speak("Welcome My friend")
    speak(uname)
    columns = shutil.get_terminal_size().columns

    print(" ".center(columns))
    print("Welcome ", uname.center(columns))
    print("  ".center(columns))

    speak("How can i Help you, Sir")
#Definition of the LISTENING() function: Our voice assistant's ability to accept commands using our system's microphone is the second-most crucial feature. Let's start by writing the code for our takeCommand() function.

#Our A.I. voice assistant will also be able to deliver a string output by taking input from us through our microphone with the aid of our takeCommand() function.

def LISTENING():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)
#Our takeCommand() method has been successfully developed. To efficiently handle our errors, we will now add a try and except block to our programme.

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language ='en-in')
        print(f"User said: {query}\n")

    except Exception as e:
        print(e)
        print("Unable to Recognize your voice.")
        return "None"

    return query

# method to accept and recognize commands given by the user
'''Defining a speak function:
The first and foremost thing for an A.I. voice assistant is to speak.
To make our bot speak, we will code a speak() function which takes audio as an input, and pronounces it, as an output.'''

#-------------------------------
def destination_language():
        print("Enter the language in which you want to convert : Ex. Hindi , English , etc.")
        print()

        # Input destination language in
        # which the user wants to translate
        to_lang = LISTENING()
        while (to_lang == "None"):
            to_lang = LISTENING()
        to_lang = to_lang.lower()
        return to_lang

to_lang = destination_language()

    # Mapping it with the code
while (to_lang not in dic):
        print("Language in which you are trying to convert is currently not available ,        please input some other language")
        print()
        to_lang = destination_language()

to_lang = dic[dic.index(to_lang)+1]


    # invoking Translator
translator = Translator()


    # Translating from src to dest
text_to_translate = translator.translate(query, dest=to_lang)

text = text_to_translate.text

    # Using Google-Text-to-Speech ie, gTTS() method
    # to speak the translated text into the
    # destination language which is stored in to_lang.
    # Also, we have given 3rd argument as False because
    # by default it speaks very slowly
speak = gTTS(text=text, lang=to_lang, slow=False)

    # Using save() method to save the translated
    # speech in capture_voice.mp3
speak.save("captured_voice.mp3")

    # Using OS module to run the translated voice.
playsound('captured_voice.mp3')
os.remove('captured_voice.mp3')

    # Printing Output
print(text)
#--------------------------------
def speak(audio):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.say(audio)
    engine.runAndWait()
    
    
#Coding the wishme() function: In this step, we will code the wishme() method, which will allow our voice assistant to wish or greet us in accordance with the current computer time.    
#The variable hour contains the integer representation of the current hour or time. In an if-else loop, we will now use this hour value.
def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour>= 0 and hour<12:
        speak("Good Morning Sir !")

    elif hour>= 12 and hour<18:
         speak("Good Afternoon Sir !")

    else:
         speak("Good Evening Sir !")


def tellDay():
    day = datetime.date.today()
    speak(calendar.day_name[day.weekday()])
    Day_dict = {1: 'Saturday', 2: 'Sunday',
                3: 'Monday', 4: 'Tuesday',
                5: 'Wednesday', 6: 'Thursday',
                7: 'Friday'}
    if day in Day_dict.keys():
        day_of_the_week = Day_dict[day]
        print(day_of_the_week)
        speak("The day is " + day_of_the_week)


def tellTime():
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    output = "Your time is " + current_time
    speak(output)
    time = str(datetime.datetime.now())
    print(time)
    hour = time[11:13]
    if hour>= 0 and hour<12:
        speak("Good Morning Sir !")
  
    elif hour>= 12 and hour<18:
        speak("Good Afternoon Sir !")  
  
    else:
        speak("Good Evening Sir !") 
    min = time[14:16]
    speak("The time is " + hour + "Hours and" + min + "Minutes")

    
def welcome():
    speak('Hello!!! I am your friend alex')

# Creating our main function: We will now create a main() function and define our own talk function inside of it.

if __name__=='__main__':
    clear = lambda: os.system('cls')
      # This Function will clean any
    clear()
      # command before execution of this python file
    wishMe()
    welcome()
    username()
 # Anything you enter into the talk() function will be fully translated into speech. Congratulations! Now that it has its own voice, our voice assistant is prepared to talk!
   

 
    while True:
        #speak("Tell me how can I help you now?")
        query = LISTENING().lower()
        if query==0:
            continue
            
            
         # All the commands said by user will be
        # stored here in 'query' and will be
        # converted to lower case for easily
        # recognition of command
        # Open youtube tab 
        elif "open youtube" in query:
            speak("Opening youtube ")
            webbrowser.open("www.youtube.com")
            continue
        # Open browser to search about anything
        elif "open browser" in query:
            speak("Opening Google Chrome ")
            webbrowser.open("www.google.com")
            continue
         # When asking assistant how are you , it responce to you and asking for yourself   
        elif 'how are you' in query:
            speak("I am fine, Thank you")
            speak("How are you, Sir")
            continue
        # When you tell the assistant that you are fine or good , " We can add more" , it apperiated to hear you fine
        elif 'fine' in query or "good" in query:
            speak("It's good to know that your fine") 
            continue
        #When you want to ask anything from google , the answer will be voiced as the 4 sentences of information at site    
        elif "from google" in query:
            speak("Checking google ")
            query = query.replace("google", "")
            result = googlesearch.summary(query, sentences=4)
            speak("According to google")
            speak(result)
            continue
        
        elif "what is today" in query:
            tellDay()
            continue

        elif "tell me the time" in query:
            tellTime()
            continue
        # When saying goodbye or exit , it exit the assistant take queries
        elif "goodbye" in query:
            speak("Good Bye Master!")
            exit()
            
        elif 'exit' in query:
            speak("Thanks for giving me your time")
            exit()
        # When you want to ask anything from wikipedia , the answer will be voiced as the 4 sentences of information at site  
        elif "from wikipedia" in query:
            speak("Checking the wikipedia ")
            query = query.replace("wikipedia", "")
            result = wikipedia.summary(query, sentences=4)
            speak("According to wikipedia")
            speak(result)
            continue
        # Telling the tourist , the assistant name 
        elif "your name" in query:
            speak("I am alex. Your Virtual Assistant!")
            continue
        
        elif 'search web' in query:
            pywhatkit.search(query)
            speak("Searching Result in Google!")
            continue

        elif 'play' in query:
            speak('playing ' + query)
            pywhatkit.playonyt(query)
            continue
          
        elif "change my name to" in query:
            query = query.replace("change my name to", "")
            assname = query
            continue
        elif "change name" in query:
            speak("What would you like to call me, Sir ")
            assname = LISTENING()
            speak("Thanks for naming me")  
            continue

        elif 'joke' in query:
            speak(pyjokes.get_joke())
            continue
            
        elif 'search' in query or 'play' in query:
             
            query = query.replace("search", "")
            query = query.replace("play", "")         
            webbrowser.open(query)
            continue
        elif 'play music' in query or "play song" in query:
            speak("Here you go with music")
            # music_dir = "G:\\Song"
            music_dir = "C:\\Users\\GAURAV\\Music"
            songs = os.listdir(music_dir)
            print(songs)   
            random = os.startfile(os.path.join(music_dir, songs[1]))
            continue
        elif "who i am" in query:
            speak("If you talk then definitely your human.")
            continue
        elif 'Egyptian Museum' in query:
            speak("opening Photo")
            museum = r"C:\\Users\\FreeComp\\Desktop\\The_Egyptian_Museum.jpg"
            os.startfile(museum)
            continue
            
        elif "don't listen" in query or "stop listening" in query:
            speak("for how much time you want to stop me from listening commands")
            a = int(LISTENING())
            time.sleep(a)
            print(a)
            continue
            
        elif "write a note" in query:
            speak("What should i write, sir")
            note = LISTENING()
            file = open('assistant.txt', 'w')
            speak("Sir, Should i include date and time")
            snfm = LISTENING()
            if 'yes' in snfm or 'sure' in snfm:
                strTime = datetime.datetime.now().strftime("% H:% M:% S")
                file.write(strTime)
                file.write(" :- ")
                file.write(note)
            else:
                file.write(note)
                continue
         
        elif "show note" in query:
            speak("Showing Notes")
            file = open("assistant.txt", "r")
            print(file.read())
            speak(file.read(6))
            continue
        # Open " Openweathermap" site and mak account on it , sign in  and get key , put key in variable to know the weather in your place
        # Google Open weather website
        # to get API of Open weather
        elif "weather" in query:
            api_key="298b0d4297869227fa5dca180b99d3ba"
            base_url="https://api.openweathermap.org/data/2.5/weather?"
            speak("what is the city name")
            city_name=LISTENING()
            complete_url=base_url+"appid="+api_key+"&q="+city_name
            response = requests.get(complete_url)
            x=response.json()
            if x["cod"]!="404":
                y=x["main"]
                current_temperature = y["temp"]
                current_humidiy = y["humidity"]
                z = x["weather"]
                weather_description = z[0]["description"]
                speak(" Temperature in kelvin unit is " +
                      str(current_temperature) +
                      "\n humidity in percentage is " +
                      str(current_humidiy) +
                      "\n description  " +
                      str(weather_description))
                print(" Temperature in kelvin unit = " +
                      str(current_temperature) +
                      "\n humidity (in percentage) = " +
                      str(current_humidiy) +
                      "\n description = " +
                      str(weather_description))
                continue
   #Source:https://medium.com/analytics-vidhya/a-guide-to-your-own-a-i-voice-assistant-using-python-17f79c94704
