# BY: Abdelraouf Hawash
# IN: 29 Apr 2023

import socket
import time
import threading

def translate(picture_path):
    
    # processing...
    # ...
    result = "still working on translation service"
    # ...

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
