# BY: Abdelraouf Hawash
# IN: 10 Feb 2023

# import libraries
import smart_tour_guide as stg
import lcd, image, time, gc

# init lcd
lcd.init()
lcd.clear() # set all pixels to zero

# audio volume
VOLUME = 10

# network connection
lcd.draw_string(15, 20, "try esp32 connect wi-fi...", lcd.WHITE)
isConnected, ifConfig = stg.connect_network(SSID = "network", PASW = "123456789")
if isConnected : lcd.draw_string(15, 40, "  {} / {}".format(ifConfig[0], ifConfig[1]), lcd.WHITE)
else : lcd.draw_string(15, 40, "no network connection", lcd.RED)

# get current time from server
lcd.draw_string(15, 80, "try to get time from server...", lcd.WHITE)
time_tuple = stg.get_time( ("192.168.1.100", 5000) )
if time_tuple is not None:
    time.set_time(time_tuple)
    lcd.draw_string(15, 100, "  {}".format(time_tuple), lcd.WHITE)
else : lcd.draw_string(15, 100, "can't get the time", lcd.RED)
time.sleep(3)

# show welcome massage on screen
lcd.display(image.Image("/sd/multimedia/images/LCD_frames/welcome.jpg"))
stg.play_wav(path= "/sd/multimedia/audios/assistant/welcome.wav", volume= VOLUME)
time.sleep(1)

# delete unneeded variables and run garbage to save memory
del isConnected, ifConfig, time_tuple
gc.collect()

# program main loop
while True:
    # show howCanIHelpYou frame
    lcd.display(image.Image("/sd/multimedia/images/LCD_frames/howCanIHelpYou.jpg") )
    stg.play_wav(path= "/sd/multimedia/audios/assistant/howCanIHelpYou.wav", volume= VOLUME)
    # record click time
    click_lastTime = time.time()
    while (time.time()- click_lastTime) < 30:

        if stg.click_num != 0:

            # isolated word recognition
            lcd.display(image.Image("/sd/multimedia/images/LCD_frames/speak_now.jpg") )
            stg.play_wav(path= "/sd/multimedia/audios/effects/record.wav", volume= 10)
            # gc.collect()
            time.sleep_ms(200)
            recognition = stg.speech_recognition()
            lcd.clear()

            if recognition == 'recognize' :
                stg.play_wav(path= "/sd/multimedia/audios/assistant/faceRecognition_srv.wav", volume= VOLUME)
                stg.face_recognition(ACCURACY= 65, volume= VOLUME)

            elif recognition == 'assistant' :
                stg.play_wav(path= "/sd/multimedia/audios/assistant/voiceAssistant_srv.wav", volume= VOLUME)
                stg.voice_assistant(addr= ("192.168.1.100", 5001), recordTime_s= 5, volume= VOLUME)

            elif recognition == 'translate' :
                stg.play_wav(path= "/sd/multimedia/audios/assistant/translation_srv.wav", volume= VOLUME)
                stg.translate(addr= ("192.168.1.100", 5002), volume= VOLUME)

            elif recognition == 'QR-code' :
                stg.play_wav(path= "/sd/multimedia/audios/assistant/QR_srv.wav", volume= VOLUME)
                stg.QR_code(addr= ("192.168.1.100", 5003), volume= VOLUME)

            elif recognition == 'capture' :
                stg.play_wav(path= "/sd/multimedia/audios/assistant/capture_srv.wav", volume= VOLUME)
                stg.capture(addr= ("192.168.1.100", 5004), volume= VOLUME)

            elif recognition == 'music' :
                stg.play_wav(path= "/sd/multimedia/audios/assistant/music_srv.wav", volume= VOLUME)
                stg.music(volume= VOLUME)

            elif recognition == 'time' :
                stg.play_wav(path= "/sd/multimedia/audios/assistant/time_srv.wav", volume= VOLUME)
                stg.show_time()

            elif recognition == 'STG' :
                stg.play_wav(path= "/sd/multimedia/audios/assistant/STG_srv.wav", volume= VOLUME)
                stg.STG_info(volume= VOLUME)

            else:
                lcd.display(image.Image("/sd/multimedia/images/LCD_frames/sad.jpg") )
                stg.play_wav(path= "/sd/multimedia/audios/assistant/dontUnderstand.wav", volume= VOLUME)

            # show howCanIHelpYou frame
            lcd.display(image.Image("/sd/multimedia/images/LCD_frames/howCanIHelpYou.jpg") )
            stg.play_wav(path= "/sd/multimedia/audios/assistant/howCanIHelpYou.wav", volume= VOLUME)
            # record click time
            click_lastTime = time.time()
            # run a garbage collection
            gc.collect()

        time.sleep_ms(100)

    # show clock in case of system is inactive
    stg.show_time()
    # run a garbage collection
    gc.collect()



