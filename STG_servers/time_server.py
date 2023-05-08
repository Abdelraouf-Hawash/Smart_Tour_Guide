# BY: Abdelraouf Hawash
# IN: 5 May 2023

import socket
import time
import threading


def receiveThread(conn):
    conn.settimeout(3)
    t = time.localtime()
    t_str = f'{t.tm_year} {t.tm_mon} {t.tm_mday} {t.tm_hour} {t.tm_min} {t.tm_sec} {t.tm_wday} {t.tm_yday}'
    # print(t_str)
    
    # send time
    conn.send(bytes(t_str, 'utf-8'))
    
    # close connection
    conn.close()
    print('connection closed')

def server():
    local_ip = ""
    local_port = 5000
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
