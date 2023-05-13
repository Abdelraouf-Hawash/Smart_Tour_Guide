# BY: Abdelraouf Hawash
# IN: 29 Apr 2023

import socket
import time
import threading
import cv2
from pyzbar import pyzbar


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
    file = open(f'./tmp/QR.jpg', 'wb')
    file.write(tmp)
    file.close()
    # read QR from image
    img = cv2.imread('./tmp/QR.jpg',0)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    barcode = pyzbar.decode(blur)
    # send code data
    if barcode:
        # barcode_data = barcode[0].data.decode("utf-8")
        # print('barcode data: ' , barcode_data)
        conn.send(barcode[0].data)
    else:
        conn.send(b'')
    # close connection
    conn.close()
    print('connection closed')

def server():
    local_ip = ""
    local_port = 5003
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
