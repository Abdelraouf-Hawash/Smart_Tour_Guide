# BY: Abdelraouf Hawash
# IN: 29 Apr 2023

import socket
import time
import threading

def receiveThread(conn):
    conn.settimeout(5)
    tmp = b''
    line = True
    while line:
        try:
            line = conn.recv(1024*4)
        except socket.timeout:
            print("socket time out!")
            conn_end = True
            break
        tmp += line

    # save received image
    print("receive end, file len:", len(tmp))
    file_name = time.strftime("%Y-%m-%d_%H:%M:%S")
    file = open(f'./saved_pictures/{file_name}.jpg', 'wb')
    file.write(tmp)
    file.close()
    # send "ok"
    conn.send(bytes("ok", 'utf-8'))
    # close connection
    conn.close()
    print('connection closed')

def server():
    local_ip = ""
    local_port = 5004
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
