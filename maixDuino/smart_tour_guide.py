
# This script contains some functions used in smart tour guide system

# BY: Abdelraouf Hawash
# IN: 10 Feb 2023

from fpioa_manager import fm
from Maix import I2S, GPIO
from board import board_info
from machine import Timer
import socket
import audio
import lcd, image, time, gc, sensor

### board settings ###

# init lcd
lcd.init()
# init audio PA
fm.register(2, fm.fpioa.GPIO1, force=True)
GPIO(GPIO.GPIO1, GPIO.OUT).value(0)
# register i2s(i2s1) pin
fm.register(34, fm.fpioa.I2S1_OUT_D1, force=True)
fm.register(35, fm.fpioa.I2S1_SCLK, force=True)
fm.register(33, fm.fpioa.I2S1_WS, force=True)
# mic initiation
fm.register(20,fm.fpioa.I2S0_IN_D0, force=True)
fm.register(18,fm.fpioa.I2S0_SCLK, force=True)
fm.register(19,fm.fpioa.I2S0_WS, force=True)
# hardware boot key initiation
fm.register(board_info.BOOT_KEY, fm.fpioa.GPIOHS0)
key_gpio = GPIO(GPIO.GPIOHS0, GPIO.IN)
click_num = 0   # key click number record
last_ticks = time.ticks_ms()

def set_clickNum(timer):
    global click_num
    click_num = timer.callback_arg()

timer0 = Timer(Timer.TIMER1, Timer.CHANNEL1, mode=Timer.MODE_ONE_SHOT, period=1200, unit=Timer.UNIT_MS, callback=set_clickNum, arg=0, start=False, priority=1, div=0)
timer1 = Timer(Timer.TIMER0, Timer.CHANNEL0, mode=Timer.MODE_ONE_SHOT, period=600, unit=Timer.UNIT_MS, callback=set_clickNum, arg=1, start=False, priority=1, div=0)

def key_callback(*_):
    global click_num
    global last_ticks
    global timer0
    global timer1
    time_diff = time.ticks_diff(time.ticks_ms(), last_ticks)
    if time_diff > 100: # avoid hardware double click error
        last_ticks = time.ticks_ms()    # record last time
        timer1.stop()   # stop last setting timer
        if time_diff> 600:
            timer1.restart()    # set click_num to 1 after 600 sec if not clicked in nex 600 ms
        else:
            click_num = 2

        timer0.stop()    # stop last resetting timer
        timer0.restart() # reset click num to 0 after 1200 ms

key_gpio.irq(key_callback, GPIO.IRQ_RISING, GPIO.WAKEUP_NOT_SUPPORT)

### some methods ###

def connect_network(SSID,PASW):
    '''
    this func used to connect wi-fi
    '''
    from network_esp32 import wifi
    if wifi.isconnected() == False:
        for i in range(5):
            try:
                wifi.reset(is_hard=True)
                print("try esp32 connect wifi...")
                wifi.connect(SSID, PASW)
                if wifi.isconnected():
                    break
            except Exception as e:
                print(e)
    return wifi.isconnected(), wifi.ifconfig()

def get_time(addr):
    '''
    this func used to get server time as a 8 tuple
    '''

    # connect to sever
    try:
        sock = socket.socket()
        sock.connect(addr)
        sock.settimeout(3)
    except Exception as e:
        print("connect error:", e)
        sock.close()
        return None

    # receive time as string
    data = b''
    while True:
        try:
            tmp = sock.recv(1)
            if len(tmp) == 0:
                raise Exception('timeout or disconnected')
            data += tmp
        except Exception as e:
            break

    # close socket
    sock.close()
    # prepare received data
    data = data.decode("utf-8")
    data = data.split() # converting to list
    ints = []
    for element in data:
        ints.append(int(element))

    if len(ints) != 8: return None
    return tuple(ints)


def play_wav(path,volume = 10):
    '''
    this func used to play audio wav
    '''

    # reset click_num variable
    global click_num
    click_num = 0
    # enable audio PA
    GPIO(GPIO.GPIO1, GPIO.OUT).value(1)
    # init i2s(I2S1)
    speaker_dev = I2S(I2S.DEVICE_1)
    # init audio
    player = audio.Audio(path= path)
    player.volume(volume)
    # read audio info
    wav_info = player.play_process(speaker_dev)
    # config i2s according to audio info
    speaker_dev.channel_config(speaker_dev.CHANNEL_1, I2S.TRANSMITTER, resolution=I2S.RESOLUTION_16_BIT, cycles=I2S.SCLK_CYCLES_32, align_mode=I2S.RIGHT_JUSTIFYING_MODE)
    speaker_dev.set_sample_rate(wav_info[1])
    # loop to play audio
    while click_num == 0:   # check if key clicked to exit
        ret = player.play()
        if ret == None:
            print("audio format error!")
            break
        elif ret == 0:
            break   # end wave
        
    player.finish()
    # disable audio PA
    GPIO(GPIO.GPIO1, GPIO.OUT).value(0)
    # reset click_num variable
    click_num = 0


def speech_recognition():
    '''
    this func used for isolated word recognition
    '''

    from speech_recognizer import isolated_word
    import sr_words
    # init i2s0
    mic_dev = I2S(I2S.DEVICE_0)
    mic_dev.channel_config(mic_dev.CHANNEL_0, mic_dev.RECEIVER, align_mode=I2S.STANDARD_MODE)
    mic_dev.set_sample_rate(16000)
    # init SR
    sr = isolated_word(dmac=2, i2s=I2S.DEVICE_0, size=20, shift=0)
    # threshold
    sr.set_threshold(0, 0, 10000)
    # set words
    words = list(sr_words.vocabulary.keys())
    all_words = []  # may be as: [word1,word1,word2,word3,word3,word3...]
    i = 0
    for w in words:
        for p in sr_words.vocabulary[w]:
            sr.set(i,p)
            all_words.append(w)
            i += 1
    
    del words, sr_words
    gc.collect()

    while click_num == 0:
        try:
            if sr.Done == sr.recognize():
                res = sr.result()
                if isinstance(res, tuple):  # to avoid None type
                    sr.__del__()
                    del sr
                    gc.collect()
                    return all_words[res[0]]
        except Exception as e: print(e)

    sr.__del__()
    del sr
    gc.collect()
    return None

def face_recognition(ACCURACY= 70, volume= 10):
    '''
    this func used for face recognition
    '''
    
    import KPU as kpu
    import fr_faces
    # load models
    task_fd = kpu.load("/sd/models/FaceRecognition/FaceDetection.smodel")
    task_ld = kpu.load("/sd/models/FaceRecognition/FaceLandmarkDetection.smodel")
    task_fe = kpu.load("/sd/models/FaceRecognition/FeatureExtraction.smodel")
    anchor = (1.889, 2.5245, 2.9465, 3.94056, 3.99987, 5.3658, 5.155437, 6.92275, 6.718375, 9.01025)  # anchor for face detect
    dst_point = [(44, 59), (84, 59), (64, 82), (47, 105), (81, 105)]  # standard face key point position
    # sensor initiation
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)   # 320X240
    # sensor.set_hmirror(1)
    # sensor.set_vflip(1)
    sensor.run(1)
    # model initiation
    kpu.init_yolo2(task_fd, 0.5, 0.3, 5, anchor)
    img_face = image.Image(size=(128, 128))
    a = img_face.pix_to_ai()
    face_names =  list(fr_faces.faces.keys())
    max_score = 0
    max_index = 0
    while click_num != 2:
        img = sensor.snapshot()
        code = kpu.run_yolo2(task_fd, img)
        if code:
            for i in code:
                # Cut face and resize to 128x128
                a = img.draw_rectangle(i.rect())
                face_cut = img.cut(i.x(), i.y(), i.w(), i.h())
                face_cut_128 = face_cut.resize(128, 128)
                a = face_cut_128.pix_to_ai()
                # Landmark for face 5 points
                fmap = kpu.forward(task_ld, face_cut_128)
                plist = fmap[:]
                le = (i.x() + int(plist[0] * i.w() - 10), i.y() + int(plist[1] * i.h()))
                re = (i.x() + int(plist[2] * i.w()), i.y() + int(plist[3] * i.h()))
                nose = (i.x() + int(plist[4] * i.w()), i.y() + int(plist[5] * i.h()))
                lm = (i.x() + int(plist[6] * i.w()), i.y() + int(plist[7] * i.h()))
                rm = (i.x() + int(plist[8] * i.w()), i.y() + int(plist[9] * i.h()))
                # align face to standard position
                src_point = [le, re, nose, lm, rm]
                T = image.get_affine_transform(src_point, dst_point)
                a = image.warp_affine_ai(img, img_face, T)
                a = img_face.ai_to_pix()
                # a = img.draw_image(img_face, (128,0))
                del (face_cut_128)
                # calculate face feature vector
                fmap = kpu.forward(task_fe, img_face)
                feature = kpu.face_encode(fmap[:])
                # calculate matching score for each face
                scores = []
                for F in face_names:
                    # get average score for the same face compared with its recorded features
                    n = len(fr_faces.faces[F])
                    score = 0
                    for f in fr_faces.faces[F]:
                        score += kpu.face_compare(f, feature)/n
                    scores.append(score)

                # calculate max score
                max_score = 0
                max_index = 0
                for k in range(len(scores)):
                    if max_score < scores[k]:
                        max_score = scores[k]
                        max_index = k
                
                # draw recognition on frame
                if max_score > ACCURACY:
                    a = img.draw_string(i.x(), i.y(), (" %s :%2.1f" % (face_names[max_index], max_score)), color=(0, 255, 0), scale= 1.25)
                else:
                    a = img.draw_string(i.x(), i.y(), " Unknown", color=(255, 0, 0), scale= 1.25)
                
                del scores, feature, fmap
                # check if single click to show face info frame
                if click_num == 1:
                    if max_score > ACCURACY:
                        gc.collect()
                        try:
                            lcd.display(image.Image("/sd/multimedia/images/LCD_frames/faces_info/{}.jpg".format(face_names[max_index]) ))
                            play_wav(path= "/sd/multimedia/audios/assistant/faces_info/{}.wav".format(face_names[max_index]), volume= volume)
                        except:
                            lcd.display(image.Image("/sd/multimedia/images/LCD_frames/problem.jpg" ))
                            play_wav(path= "/sd/multimedia/audios/assistant/problem.wav", volume= volume)
                        time.sleep(2)

        a = lcd.display(img)
        gc.collect()

    a = kpu.deinit(task_fe)
    a = kpu.deinit(task_ld)
    a = kpu.deinit(task_fd)
    gc.collect()

def get_face_features():
    '''
    this func used to get face feature from cam
    can be used from terminal as:  import smart_tour_guide as stg; stg.get_face_features()
    press boot key single click to get features or double key to exit
    '''
    
    import lcd, image, sensor
    import KPU as kpu
    import gc
    global click_num
    # load models
    task_fd = kpu.load("/sd/models/FaceRecognition/FaceDetection.smodel")
    task_ld = kpu.load("/sd/models/FaceRecognition/FaceLandmarkDetection.smodel")
    task_fe = kpu.load("/sd/models/FaceRecognition/FeatureExtraction.smodel")
    anchor = (1.889, 2.5245, 2.9465, 3.94056, 3.99987, 5.3658, 5.155437, 6.92275, 6.718375, 9.01025)  # anchor for face detect
    dst_point = [(44, 59), (84, 59), (64, 82), (47, 105), (81, 105)]  # standard face key point position
    # sensor initiation
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)   # 320X240
    # sensor.set_hmirror(1)
    # sensor.set_vflip(1)
    sensor.run(1)
    # model initiation
    kpu.init_yolo2(task_fd, 0.5, 0.3, 5, anchor)
    img_face = image.Image(size=(128, 128))
    a = img_face.pix_to_ai()
    while click_num != 2 :
        img = sensor.snapshot()
        code = kpu.run_yolo2(task_fd, img)
        if code:
            for i in code:
                # Cut face and resize to 128x128
                a = img.draw_rectangle(i.rect())
                face_cut = img.cut(i.x(), i.y(), i.w(), i.h())
                face_cut_128 = face_cut.resize(128, 128)
                a = face_cut_128.pix_to_ai()
                # Landmark for face 5 points
                fmap = kpu.forward(task_ld, face_cut_128)
                plist = fmap[:]
                le = (i.x() + int(plist[0] * i.w() - 10), i.y() + int(plist[1] * i.h()))
                re = (i.x() + int(plist[2] * i.w()), i.y() + int(plist[3] * i.h()))
                nose = (i.x() + int(plist[4] * i.w()), i.y() + int(plist[5] * i.h()))
                lm = (i.x() + int(plist[6] * i.w()), i.y() + int(plist[7] * i.h()))
                rm = (i.x() + int(plist[8] * i.w()), i.y() + int(plist[9] * i.h()))
                a = img.draw_circle(le[0], le[1], 4)
                a = img.draw_circle(re[0], re[1], 4)
                a = img.draw_circle(nose[0], nose[1], 4)
                a = img.draw_circle(lm[0], lm[1], 4)
                a = img.draw_circle(rm[0], rm[1], 4)
                # align face to standard position
                src_point = [le, re, nose, lm, rm]
                T = image.get_affine_transform(src_point, dst_point)
                a = image.warp_affine_ai(img, img_face, T)
                a = img_face.ai_to_pix()
                del (face_cut_128)
                # calculate face feature vector
                fmap = kpu.forward(task_fe, img_face)
                feature = kpu.face_encode(fmap[:])
                # check if single click to print face features
                if click_num == 1 :
                    print(feature)
                    click_num = 0
        a = lcd.display(img)
        gc.collect()

    a = kpu.deinit(task_fe)
    a = kpu.deinit(task_ld)
    a = kpu.deinit(task_fd)
    gc.collect()


def info_frame(info = '', x= 10, y=20, scale= 1.6, bg= (200, 200, 200), fg= (255, 0, 0)):
    '''
    this func generate info frame
    '''
    import image, lcd
    loading = image.Image(size=(lcd.width(), lcd.height()))
    loading.draw_rectangle((0, 0, lcd.width(), lcd.height()), fill=True, color=bg)
    loading.draw_string(x, y, info, color=fg, scale= scale, mono_space=0)
    return loading


def sendFile_receiveText(path, addr):
    '''
    this function sends file to server with address
    return with received text and return with None if there is a connection error
    '''

    try:
        sock = socket.socket()
        sock.connect(addr)
        sock.settimeout(5)
    except Exception as e:
        print("connect error:", e)
        sock.close()
        return None

    file = open(path,'rb')
    line = file.read(64)
    err = 0
    while line:
        if err >=10:
            print("socket broken")
            file.close()
            sock.close()
            return None
        try:
            send_len = sock.send(line)
            if send_len == 0:
                raise Exception("send fail")
        except OSError as e:
            if e.args[0] == 128:
                print("connection closed")
                break
        except Exception as e:
            print("send fail:", e)
            time.sleep(1)
            err += 1
            continue
        line = file.read(64)

    file.close()
    # receive text
    data = b''
    while True:
        try:
            tmp = sock.recv(1)
            if len(tmp) == 0:
                raise Exception('timeout or disconnected')
            data += tmp
        except Exception as e:
            break
    data = data.decode("utf-8")
    # close socket
    sock.close()
    return data

def voice_assistant(addr, recordTime_s= 5, volume= 10):
    '''
    this func used to record voice then send it to remote server and get assistant result
    '''

    global click_num
    # user setting
    sample_rate   = 2000
    # default settings
    sample_points = 2048
    wav_ch        = 2
    frame_cnt = recordTime_s*sample_rate//sample_points
    # init i2s0
    mic_dev = I2S(I2S.DEVICE_0)
    mic_dev.channel_config(mic_dev.CHANNEL_0, mic_dev.RECEIVER, align_mode=I2S.STANDARD_MODE)
    mic_dev.set_sample_rate(sample_rate)
    # show voice assistant frame
    lcd.display(image.Image("/sd/multimedia/images/LCD_frames/voice_assistant.jpg"))

    while click_num != 2:
        
        if click_num == 1:
            lcd.display(image.Image("/sd/multimedia/images/LCD_frames/speak_now.jpg"))
            play_wav(path= "/sd/multimedia/audios/effects/pop1.wav", volume= volume)
            # start recording
            recorder = audio.Audio(path="/sd/tmp/record.wav", is_create=True, samplerate=sample_rate)
            queue = []
            for i in range(frame_cnt):
                tmp = mic_dev.record(sample_points*wav_ch)
                if len(queue) > 0:
                    ret = recorder.record(queue[0])
                    queue.pop(0)
                mic_dev.wait_record()
                queue.append(tmp)
            
            # recording end
            recorder.finish()
            #del recorder, queue
            play_wav(path= "/sd/multimedia/audios/effects/pop2.wav", volume= volume)
            # send wav
            lcd.display(image.Image("/sd/multimedia/images/LCD_frames/loading.jpg"))
            ret = sendFile_receiveText(path= "/sd/tmp/record.wav", addr= addr)
            if ret is not None :
                lcd.display(info_frame(info = str(ret)))
                while click_num == 0 : time.sleep_ms(100)
                click_num = 0
            else:
                lcd.display(image.Image("/sd/multimedia/images/LCD_frames/problem.jpg" ))
                play_wav(path= "/sd/multimedia/audios/assistant/problem.wav", volume= volume)
                time.sleep(1)

            # show voice assistant frame
            lcd.display(image.Image("/sd/multimedia/images/LCD_frames/voice_assistant.jpg"))
            gc.collect()
        
        time.sleep_ms(100)
        

def translate(addr, volume= 10):
    '''
    this func used to take a picture from camera then process it on remote server to translate and show result
    '''
    
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.VGA)    # 640X480
    while click_num != 2:
        img = sensor.snapshot()
        if click_num == 1:
            lcd.draw_string(120,120,"processing...",lcd.BLACK,lcd.WHITE)
            play_wav(path= "/sd/multimedia/audios/effects/take_photo.wav", volume= volume)
            img.save("/sd/tmp/picture.jpg")
            ret = sendFile_receiveText("/sd/tmp/picture.jpg", addr)
            if ret is not None :
                lcd.display(info_frame(info = str(ret)))
                while click_num == 0 : time.sleep_ms(100)
            else:
                lcd.display(image.Image("/sd/multimedia/images/LCD_frames/problem.jpg" ))
                play_wav(path= "/sd/multimedia/audios/assistant/problem.wav", volume= volume)

            time.sleep(1)

        else:
            lcd.display(img.resize(320, 240))


def QR_code(addr, volume= 10):
    '''
    this func used to take a picture from camera then process it on remote server and show result
    '''

    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.VGA)    # 640X480
    while click_num != 2:
        img = sensor.snapshot()
        if click_num == 1:
            lcd.draw_string(120,120,"processing...",lcd.BLACK,lcd.WHITE)
            play_wav(path= "/sd/multimedia/audios/effects/take_photo.wav", volume= volume)
            img.save("/sd/tmp/QR.jpg")
            ret = sendFile_receiveText("/sd/tmp/QR.jpg", addr)
            if ret is not None :
                lcd.display(info_frame(info = str(ret)))
                while click_num == 0 : time.sleep_ms(100)
            else:
                lcd.display(image.Image("/sd/multimedia/images/LCD_frames/problem.jpg" ))
                play_wav(path= "/sd/multimedia/audios/assistant/problem.wav", volume= volume)

            time.sleep(1)

        else:
            lcd.display(img.resize(320, 240))


def capture(addr, volume= 10):
    '''
    this func used to take a picture from camera then save it on remote server
    '''
    
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.VGA)    # 640X480
    while click_num != 2:
        img = sensor.snapshot()
        if click_num == 1:
            lcd.draw_string(130,120,"Saving...",lcd.BLACK,lcd.WHITE)
            play_wav(path= "/sd/multimedia/audios/effects/take_photo.wav", volume= volume)
            img.save("/sd/tmp/picture.jpg")
            ret = sendFile_receiveText("/sd/tmp//picture.jpg", addr)
            if ret is None:
                lcd.display(image.Image("/sd/multimedia/images/LCD_frames/problem.jpg" ))
                play_wav(path= "/sd/multimedia/audios/assistant/problem.wav", volume= volume)
            
            time.sleep(1)

        else:
            lcd.display(img.resize(320, 240))


def music(volume= 10):
    '''
    this func used to play music
    '''

    import os
    # show music frame
    lcd.display(image.Image("/sd/multimedia/images/LCD_frames/music.jpg" ))
    # get music list 
    music_list = os.listdir("/sd/multimedia/audios/music")
    # check if there are music
    if len(music_list) == 0 :
        lcd.display(image.Image("/sd/multimedia/images/LCD_frames/problem.jpg" ))
        play_wav(path= "/sd/multimedia/audios/assistant/problem.wav", volume= volume)
        return
    
    index = 0
    while click_num != 2:

        if click_num == 1:
            if index == len(music_list): index = 0  # loop again in music list
            music_path = "/sd/multimedia/audios/music/{}".format(music_list[index])
            # print(music_path)
            play_wav(path= music_path, volume= volume)
            # get next music index
            index += 1

        time.sleep_ms(100)


def show_time():
    '''
    this func used to show local time on screen
    '''

    while click_num == 0:
        t = time.localtime()
        time_str = "    {}:{}\n\n{}/{}/{}".format(t[3], t[4], t[2], t[1], t[0])
        lcd.display( info_frame(info= time_str, x=90, y=70, scale=3, bg= (0,0,0), fg=(255,255,255)) ) 
        time.sleep_ms(100)


def STG_info(volume= 10):
    '''
    this func used to give user information about STG team
    '''

    # show STG_team frame
    lcd.display(image.Image("/sd/multimedia/images/LCD_frames/STG_team.jpg") )
    play_wav(path= "/sd/multimedia/audios/assistant/STG_team_info.wav", volume= volume)
    # stop until key clicked
    while click_num == 0 : time.sleep_ms(100)
    # display STG telegram QR-code frame
    lcd.display(image.Image("/sd/multimedia/images/LCD_frames/STG_telegram.jpg") )
    time.sleep(1)
    # stop until key clicked
    while click_num == 0 : time.sleep_ms(100)

