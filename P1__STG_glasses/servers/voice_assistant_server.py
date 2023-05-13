# BY: Alaa Taha Elmaria & Abdelraouf Hawash
# IN: 29 Apr 2023

import socket
import time
import threading
import speech_recognition as sr
import wikipedia
import textwrap


def search_wikipedia(query):

    try:
        page = wikipedia.page(query)
        content = page.content
        # wrapping content of maximum 35 characters for each line
        lines = textwrap.wrap(content, 34)
        # Limit to 10 lines
        if len(lines) > 10:
            lines = "\n".join(lines[:10])
        else:
            lines = "\n".join(lines)
        return lines
    except wikipedia.exceptions.DisambiguationError as e:
        return "Multiple results found. \nPlease provide a more specific query."
    
    except wikipedia.exceptions.PageError:
        return "No results found. \nPlease try a different query."
    

def voice_assistant(voice_path):
    
    # audio object 
    recognizer = sr.Recognizer()
    # read audio object and transcribe
    audio = sr.AudioFile(voice_path)
    with audio as source:
        audio = recognizer.record(source)  
    # Processing audio to get text
    try:
        text = recognizer.recognize_google(audio)
    except:
        return "I don't understand!"
    # Search Wikipedia for the user's query
    return search_wikipedia(text)


def receiveThread(conn):
    conn.settimeout(15)
    tmp = b''
    line = True
    while line:
        try:
            line = conn.recv(1024*4)
        except socket.timeout:
            # print("socket time out!")
            break
        tmp += line

    # save received image
    # print("receive end, file len:", len(tmp))
    path = './tmp/record.wav'
    file = open(path, 'wb')
    file.write(tmp)
    file.close()
    # get voice assistant result
    assistant_result = voice_assistant(path)
    print(assistant_result)
    # sending result
    try:
        conn.send(bytes(assistant_result, 'utf-8'))
    except:
        print("connection error")
        conn.close()
        return
    # close connection
    conn.close()
    print('connection closed')

def server():
    local_ip = ""
    local_port = 5001
    ip_port = (local_ip, local_port)
    sk = socket.socket()
    sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sk.bind(ip_port)
    sk.listen(50)
    print("accept now,wait for clients")
    while True:
        conn, addr = sk.accept()
        print("connected to client with ip:", addr[0])
        t = threading.Thread(target=receiveThread, args=(conn,))
        t.Daemon = True
        t.start()

server()
