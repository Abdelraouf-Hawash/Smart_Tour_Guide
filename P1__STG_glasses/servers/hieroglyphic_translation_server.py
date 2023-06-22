# BY: Mahmoud Hamza Abdella  & Abdelraouf Hawash
# IN: 29 Apr 2023

import cv2
import socket
import time
import threading
import sys
import os

# Get the current directory
current_dir = os.getcwd()

# Move two levels up
parent_dir = os.path.dirname(os.path.dirname(current_dir))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from P2__pharaonic_languages_translation.after import run_2
#return to current directory

sys.path.append(current_dir)

def translate(picture_path):  
    # processing...
    
    result =run_2(source=picture_path) 

    return result

def receiveThread(conn):
    conn.settimeout(3)
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
    path = './tmp/picture.jpg'
    file = open(path, 'wb')
    file.write(tmp)
    file.close()
    # get translation result
    translation_result = translate(path)
    if translation_result:
        conn.send(bytes(translation_result, 'utf-8'))
    else:
        conn.send(b'')
    # close connection
    conn.close()
    print('connection closed')

def server():
    local_ip = ""
    local_port = 5002
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
