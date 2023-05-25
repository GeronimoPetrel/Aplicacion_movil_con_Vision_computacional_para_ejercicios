from flet import *
import cv2
import time
import os
import base64
import mediapipe as mp # Debe instalarse mediante pip install mediapipe
import numpy as np
import time
import wave
import pyaudio
import winsound

cap=cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
def calcular_angulo(a,b,c):
    a=np.array(a) #primer punto
    b=np.array(b) #punto medio
    c=np.array(c) #punto final

    radianes=np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
    angulo =np.abs(radianes*180/np.pi) 

    if angulo > 180:
      angulo=360-angulo

    return angulo
#==============================================
#=============================================

def play_audio(file_path):
    # Open the .wav file
    wav = wave.open(file_path, 'rb')

    # Create an instance of the PyAudio class
    audio = pyaudio.PyAudio()

    # Open a stream
    stream = audio.open(
        format=audio.get_format_from_width(wav.getsampwidth()),
        channels=wav.getnchannels(),
        rate=wav.getframerate(),
        output=True
    )

    # Read and play the audio data
    chunk_size = 1024
    data = wav.readframes(chunk_size)
    while len(data) > 0:
        stream.write(data)
        data = wav.readframes(chunk_size)

    # Cleanup
    stream.stop_stream()
    stream.close()
    wav.close()
    audio.terminate()
#==============================================
#==============================================
n=0
ej=0
up=0
mid=0
down=0
angside=0
bot=0
done=0
start_time=-3
AltArray=[]
angsideArray=[]
file_path1 = 'Nowbell.wav'
file_path2 = 'Error2.wav'
file_path3 = 'Correct.wav'

def situp(limMin, limMax):
    
        global up, mid, down, bot, done, start_time
        chan=0
        if ang_rd>limMax:   
            if up==0:
                chan=1
            up=1   

        elif limMax>=ang_rd and ang_rd>=limMin:     
            if mid==0:
                chan=1
            mid=1
            
        else: 
            if down==0:
                chan=1
            down=1
            
        end_time=time.time()

        gap=(end_time-start_time)

        if chan==1:
            if up==1 and mid==0 and done==0:
                if gap<3:
                    up=0
                else:
                    print('sigue bajando')
                    #play_audio(file_path1)  
                    winsound.PlaySound(file_path1,winsound.SND_ASYNC) 
            elif up==1 and mid==1 and done==0:
                print('angulo correcto')
                #play_audio(file_path1)  
                winsound.PlaySound(file_path1,winsound.SND_ASYNC)
                up=0
                done=1
            elif mid==1 and down==1 and bot==0:
                print('bajaste demasiado')
                #play_audio(file_path2)  
                winsound.PlaySound(file_path2,winsound.SND_ASYNC)
                mid=0
                bot=1 
            elif mid==1 and down==1 and bot==1:
                print('Volviste al angulo correcto')
                #play_audio(file_path1)  
                winsound.PlaySound(file_path1,winsound.SND_ASYNC)
                down=0
                bot=0
            
            elif  mid == 1 and up==1 and done==1:
                print('Excelente trabajo')
                #play_audio(file_path3)
                winsound.PlaySound(file_path3,winsound.SND_ASYNC)
                up = 0
                mid = 0
                down = 0
                bot = 0
                done = 0
                start_time=time.time()

def deadlift(ElimMin, ElimMax):
    
    global mid, down, bot, done, start_time, n

    chan=0

    if ElimMax>=ang_esp and ang_esp>=ElimMin:  
        if mid==0:
            chan=1
        mid=1
        down=0
        
        
    elif ElimMin>ang_esp:
        if down==0:
            chan=1
        down=1
        mid=0

    else:
        mid=0

    end_time=time.time()

    if ang_cad<150:
        n=1

    if ang_cad>=150 and n==1:
        done=1  

    gap=(end_time-start_time)

    if chan==1:

        if mid==1 and done==0:  
            if gap<3:
                mid=0
            else: 
                print('Posición adecuada puedes iniciar')
                
                
                winsound.PlaySound(file_path1,winsound.SND_ASYNC)


        elif down==1:
            print('Inclina un poco menos la espalda')

            #play_audio(file_path2)
            winsound.PlaySound(file_path2,winsound.SND_ASYNC)

        elif mid==1 and done==1:
            print('Excelente trabajo')
            #play_audio(file_path3) 
            winsound.PlaySound(file_path3,winsound.SND_ASYNC)  
            start_time=time.time()
            down=0
            mid=0
            done=0  



def main (page: Page):


    def shrink(e):
        page_2.controls[0].width =10
        page_2.controls[0].scale=transform.Scale(
            1,alignment=alignment.center)
        page_2.update()
        

    def restore(e):
        page_2.controls[0].width =400
        page_2.controls[0].scale=transform.Scale(
            1,alignment=alignment.center)
        page_2.update()

    def stop(e):
        cap.release()
        cv2.destroyAllWindows()
        video_90_view.update()
    def play(e):
        stop==False
        global cap
        cap=cv2.VideoCapture(0)
        
        imagen_90()
        video_90_view.update()

    class imagen_90(UserControl):
        
        def __init__(self):
            super().__init__()
            self.timer = 0
            self.cap = cv2.VideoCapture(0) 

        
        def did_mount(self):
            self.update_timer()
        
        def update_timer(self):
            _, frame = cap.read()
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while cap.isOpened():
                    _, frame = cap.read() # guardar variables de la camara 
                    frame =cv2.flip(frame,1) # Frame es la variable donde se guarda las imagenes del video, esta linea pone el video tipo espejo 
                    alto, ancho,_ = frame.shape # alto y ancho del video  
                    frame_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Se pasa el video de BGR a RGB para poder usar mediapipe 
                    
                    #===se obtienen los marcadores===
                    resul = pose.process(frame_rgb)

                    if resul.pose_landmarks is not None:
                        '''
                        ================ número para marca ============================
                        (16)muñeca der. (14)codo der. (12)hombo der. (24)cadeera der. 
                        (26)rodilla der.(28)tobillo der. (32)punta pie der.
                        (15)muñeca izq. (13)codo izq. (11)hombo izq. (23)cadeera izq. 
                        (25)rodilla izq.(27)tobillo izq. (31)punta pie izq.
                        ================================================================
                        '''
                        marca=resul.pose_landmarks.landmark

                        #punto flotante
                        imx=int(marca[11].x*ancho)
                        imy=int(marca[23].y*alto)
                        im=np.array([imx,imy])


                        #puntos de codo derecho[cd]
                        cdx=int(marca[14].x *ancho)
                        cdy=int(marca[14].y *alto)
                        cd=np.array([cdx,cdy]) # Este se utiliza para los calculos de angulo


                        #puntos de codo izquierdo[cd]
                        cix=int(marca[13].x *ancho)
                        ciy=int(marca[13].y *alto)
                        ci=np.array([cix,ciy]) # Este se utiliza para los calculos de angulo

                        #puntos de muñeca izquierda[md]
                        mix=int(marca[15].x *ancho)
                        miy=int(marca[15].y *alto)
                        mi=np.array([mix,miy]) # Este se utiliza para los calculos de angulo
                        
                        #puntos de muñeca derecha[md]
                        mdx=int(marca[16].x *ancho)
                        mdy=int(marca[16].y *alto)
                        md=np.array([mdx,mdy]) # Este se utiliza para los calculos de angulo
                        #========================

                        #Puntos de talon derecho[tald]
                        taldx=int(marca[28].x *ancho)
                        taldy=int(marca[28].y *alto)
                        tald=np.array([taldx,taldy]) # Este se utiliza para los calculos de angulo
                    
                        #Puntos de talon izquierdo[tali]
                        talix=int(marca[27].x *ancho)
                        taliy=int(marca[27].y *alto)
                        tali=np.array([talix,taliy]) # Este se utiliza para los calculos de angulo

                        #Puntos de puntapie derecho[tald]
                        pudx=int(marca[32].x *ancho)
                        pudy=int(marca[32].y *alto)
                        pud=np.array([pudx,pudy]) # Este se utiliza para los calculos de angulo
                    
                        #Puntos de puntapie izquierdo[tali]
                        puix=int(marca[31].x *ancho)
                        puiy=int(marca[31].y *alto)
                        pui=np.array([puix,puiy]) # Este se utiliza para los calculos de angulo


                        #Puntos de hombro derecho[hd]
                        hdx=int(marca[12].x *ancho)
                        hdy=int(marca[12].y *alto)
                        hd=np.array([hdx,hdy]) # Este se utiliza para los calculos de angulo

                        #Puntos de hombro izquierdo[hi]
                        hix=int(marca[11].x *ancho)
                        hiy=int(marca[11].y *alto)
                        hi=np.array([hix,hiy]) # Este se utiliza para los calculos de angulo

                        #puntos de cadera derecha[cad]
                        cadx=int(marca[24].x *ancho)
                        cady=int(marca[24].y *alto)
                        cad=np.array([cadx,cady]) # Este se utiliza para los calculos de angulo

                        #puntos de cadera izquiera[cai]
                        caix=int(marca[23].x *ancho)
                        caiy=int(marca[23].y *alto)
                        cai=np.array([caix,caiy]) # Este se utiliza para los calculos de angulo

                        #puntos de rodilla derecha[rd]
                        rdx=int(marca[26].x *ancho)
                        rdy=int(marca[26].y *alto)
                        rd=np.array([rdx,rdy]) # Este se utiliza para los calculos de angulo

                        #puntos de rodilla derecha[rd]
                        rix=int(marca[25].x *ancho)
                        riy=int(marca[25].y *alto)
                        ri=np.array([rix,riy]) # Este se utiliza para los calculos de angulo

                        #puntos de tobillo derecho[td]
                        tdx=int(marca[28].x *ancho)
                        tdy=int(marca[28].y *alto)
                        td=np.array([tdx,tdy]) # Este se utiliza para los calculos de angulo

                        #puntos de tobillo izquierdo[ti]
                        tix=int(marca[27].x *ancho)
                        tiy=int(marca[27].y *alto)
                        ti=np.array([tix,tiy]) # Este se utiliza para los calculos de angulo
                        global ang_cad,ang_rd,ang_hd,ang_cd,ang_cai,ang_ri,ang_esp
                        ang_cad=calcular_angulo(hd,cad,rd) #codo
                        ang_rd=calcular_angulo(cad,rd,td) #rodilla
                        ang_hd=calcular_angulo(cd,hd,cad) #hombro
                        ang_cd=calcular_angulo(md,cd,hd) #cadera
                        ang_cai=calcular_angulo(hi,cai,ri)
                        ang_ri=calcular_angulo(cai,ri,ti)
                        ang_esp=calcular_angulo(hd,cad,im)

                        cv2.line(frame,(cadx,cady),(hdx,hdy),(0,255,0),3)

                        cv2.line(frame,(cdx,cdy),(mdx,mdy),(0,255,0),3) 
                        cv2.line(frame,(cdx,cdy),(hdx,hdy),(0,255,0),3)  
                        limMin=90 #90 grados de flexión
                        limMax=110 # 70 grados de flexión
                        situp(limMin, limMax)
                        if limMin < ang_rd < limMax:
                            cv2.line(frame,(tdx,tdy),(rdx,rdy),(0,255,0),3) 
                            cv2.line(frame,(rdx,rdy),(cadx,cady),(0,255,0),3)
                        else:           
                            cv2.line(frame,(tdx,tdy),(rdx,rdy),(0,0,255),3) 
                            cv2.line(frame,(rdx,rdy),(cadx,cady),(0,0,255),3)
                        
                        video_90_view.update()
                        _,im_arr=cv2.imencode('.png',frame)
                        im_b64= base64.b64encode(im_arr)
                        self.img.src_base64=im_b64.decode('utf-8')
                        self.update()
        
        def build(self):
            self.img=Image(width=400,height=300,
                border_radius=border_radius.all(20)
                )
            
            return Column([
                self.img,
                Row([
                    Text('funciono',size=30,weight='bold',
                         color='white')
                ])
            ])
        
    class imagen_depp(UserControl):

        def __init__(self):
            super().__init__()
            self.timer = 0
            self.cap = cv2.VideoCapture(0) 

        
        def did_mount(self):
            self.update_timer()
        
        def update_timer(self):
            _, frame = cap.read()
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while cap.isOpened():
                    _, frame = cap.read() # guardar variables de la camara 
                    frame =cv2.flip(frame,1) # Frame es la variable donde se guarda las imagenes del video, esta linea pone el video tipo espejo 
                    alto, ancho,_ = frame.shape # alto y ancho del video  
                    frame_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Se pasa el video de BGR a RGB para poder usar mediapipe 
                    
                    #===se obtienen los marcadores===
                    resul = pose.process(frame_rgb)

                    if resul.pose_landmarks is not None:
                        '''
                        ================ número para marca ============================
                        (16)muñeca der. (14)codo der. (12)hombo der. (24)cadeera der. 
                        (26)rodilla der.(28)tobillo der. (32)punta pie der.
                        (15)muñeca izq. (13)codo izq. (11)hombo izq. (23)cadeera izq. 
                        (25)rodilla izq.(27)tobillo izq. (31)punta pie izq.
                        ================================================================
                        '''
                        marca=resul.pose_landmarks.landmark

                        #punto flotante
                        imx=int(marca[11].x*ancho)
                        imy=int(marca[23].y*alto)
                        im=np.array([imx,imy])


                        #puntos de codo derecho[cd]
                        cdx=int(marca[14].x *ancho)
                        cdy=int(marca[14].y *alto)
                        cd=np.array([cdx,cdy]) # Este se utiliza para los calculos de angulo


                        #puntos de codo izquierdo[cd]
                        cix=int(marca[13].x *ancho)
                        ciy=int(marca[13].y *alto)
                        ci=np.array([cix,ciy]) # Este se utiliza para los calculos de angulo

                        #puntos de muñeca izquierda[md]
                        mix=int(marca[15].x *ancho)
                        miy=int(marca[15].y *alto)
                        mi=np.array([mix,miy]) # Este se utiliza para los calculos de angulo
                        
                        #puntos de muñeca derecha[md]
                        mdx=int(marca[16].x *ancho)
                        mdy=int(marca[16].y *alto)
                        md=np.array([mdx,mdy]) # Este se utiliza para los calculos de angulo
                        #========================

                        #Puntos de talon derecho[tald]
                        taldx=int(marca[28].x *ancho)
                        taldy=int(marca[28].y *alto)
                        tald=np.array([taldx,taldy]) # Este se utiliza para los calculos de angulo
                    
                        #Puntos de talon izquierdo[tali]
                        talix=int(marca[27].x *ancho)
                        taliy=int(marca[27].y *alto)
                        tali=np.array([talix,taliy]) # Este se utiliza para los calculos de angulo

                        #Puntos de puntapie derecho[tald]
                        pudx=int(marca[32].x *ancho)
                        pudy=int(marca[32].y *alto)
                        pud=np.array([pudx,pudy]) # Este se utiliza para los calculos de angulo
                    
                        #Puntos de puntapie izquierdo[tali]
                        puix=int(marca[31].x *ancho)
                        puiy=int(marca[31].y *alto)
                        pui=np.array([puix,puiy]) # Este se utiliza para los calculos de angulo


                        #Puntos de hombro derecho[hd]
                        hdx=int(marca[12].x *ancho)
                        hdy=int(marca[12].y *alto)
                        hd=np.array([hdx,hdy]) # Este se utiliza para los calculos de angulo

                        #Puntos de hombro izquierdo[hi]
                        hix=int(marca[11].x *ancho)
                        hiy=int(marca[11].y *alto)
                        hi=np.array([hix,hiy]) # Este se utiliza para los calculos de angulo

                        #puntos de cadera derecha[cad]
                        cadx=int(marca[24].x *ancho)
                        cady=int(marca[24].y *alto)
                        cad=np.array([cadx,cady]) # Este se utiliza para los calculos de angulo

                        #puntos de cadera izquiera[cai]
                        caix=int(marca[23].x *ancho)
                        caiy=int(marca[23].y *alto)
                        cai=np.array([caix,caiy]) # Este se utiliza para los calculos de angulo

                        #puntos de rodilla derecha[rd]
                        rdx=int(marca[26].x *ancho)
                        rdy=int(marca[26].y *alto)
                        rd=np.array([rdx,rdy]) # Este se utiliza para los calculos de angulo

                        #puntos de rodilla derecha[rd]
                        rix=int(marca[25].x *ancho)
                        riy=int(marca[25].y *alto)
                        ri=np.array([rix,riy]) # Este se utiliza para los calculos de angulo

                        #puntos de tobillo derecho[td]
                        tdx=int(marca[28].x *ancho)
                        tdy=int(marca[28].y *alto)
                        td=np.array([tdx,tdy]) # Este se utiliza para los calculos de angulo

                        #puntos de tobillo izquierdo[ti]
                        tix=int(marca[27].x *ancho)
                        tiy=int(marca[27].y *alto)
                        ti=np.array([tix,tiy]) # Este se utiliza para los calculos de angulo
                        global ang_cad,ang_rd,ang_hd,ang_cd,ang_cai,ang_ri,ang_esp
                        ang_cad=calcular_angulo(hd,cad,rd) #codo
                        ang_rd=calcular_angulo(cad,rd,td) #rodilla
                        ang_hd=calcular_angulo(cd,hd,cad) #hombro
                        ang_cd=calcular_angulo(md,cd,hd) #cadera
                        ang_cai=calcular_angulo(hi,cai,ri)
                        ang_ri=calcular_angulo(cai,ri,ti)
                        ang_esp=calcular_angulo(hd,cad,im)
                        cv2.line(frame,(cadx,cady),(hdx,hdy),(0,255,0),3)
                        cv2.line(frame,(tdx,tdy),(rdx,rdy),(0,255,0),3) 
                        cv2.line(frame,(rdx,rdy),(cadx,cady),(0,255,0),3)
                        cv2.line(frame,(cdx,cdy),(mdx,mdy),(0,255,0),3) 
                        cv2.line(frame,(cdx,cdy),(hdx,hdy),(0,255,0),3) 
                        limMin=60 # 120 grados de flexión
                        limMax=90 # 90 grados de flexión
                        situp(limMin, limMax)

                        if limMin < ang_rd < limMax:
                            cv2.line(frame,(tdx,tdy),(rdx,rdy),(0,255,0),3) 
                            cv2.line(frame,(rdx,rdy),(cadx,cady),(0,255,0),3)
                        else:           
                            cv2.line(frame,(tdx,tdy),(rdx,rdy),(0,0,255),3) 
                            cv2.line(frame,(rdx,rdy),(cadx,cady),(0,0,255),3)
                        
                        video_deep_view.update()
                        _,im_arr=cv2.imencode('.png',frame)
                        im_b64= base64.b64encode(im_arr)
                        self.img.src_base64=im_b64.decode('utf-8')
                        self.update()
        
        def build(self):
            self.img=Image(width=400,height=300,
                border_radius=border_radius.all(20)
                )
            
            return Column([
                self.img,
                Row([
                    Text('funciono',size=30,weight='bold',
                         color='white')
                ])
            ])
        
    class imagen_dead(UserControl):

        def __init__(self):
            super().__init__()
            self.timer = 0
            self.cap = cv2.VideoCapture(0) 

        
        def did_mount(self):
            self.update_timer()
        
        def update_timer(self):
            _, frame = cap.read()
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while cap.isOpened():
                    _, frame = cap.read() # guardar variables de la camara 
                    frame =cv2.flip(frame,1) # Frame es la variable donde se guarda las imagenes del video, esta linea pone el video tipo espejo 
                    alto, ancho,_ = frame.shape # alto y ancho del video  
                    frame_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Se pasa el video de BGR a RGB para poder usar mediapipe 
                    
                    #===se obtienen los marcadores===
                    resul = pose.process(frame_rgb)

                    if resul.pose_landmarks is not None:
                        '''
                        ================ número para marca ============================
                        (16)muñeca der. (14)codo der. (12)hombo der. (24)cadeera der. 
                        (26)rodilla der.(28)tobillo der. (32)punta pie der.
                        (15)muñeca izq. (13)codo izq. (11)hombo izq. (23)cadeera izq. 
                        (25)rodilla izq.(27)tobillo izq. (31)punta pie izq.
                        ================================================================
                        '''
                        marca=resul.pose_landmarks.landmark

                        #punto flotante
                        imx=int(marca[11].x*ancho)
                        imy=int(marca[23].y*alto)
                        im=np.array([imx,imy])


                        #puntos de codo derecho[cd]
                        cdx=int(marca[14].x *ancho)
                        cdy=int(marca[14].y *alto)
                        cd=np.array([cdx,cdy]) # Este se utiliza para los calculos de angulo


                        #puntos de codo izquierdo[cd]
                        cix=int(marca[13].x *ancho)
                        ciy=int(marca[13].y *alto)
                        ci=np.array([cix,ciy]) # Este se utiliza para los calculos de angulo

                        #puntos de muñeca izquierda[md]
                        mix=int(marca[15].x *ancho)
                        miy=int(marca[15].y *alto)
                        mi=np.array([mix,miy]) # Este se utiliza para los calculos de angulo
                        
                        #puntos de muñeca derecha[md]
                        mdx=int(marca[16].x *ancho)
                        mdy=int(marca[16].y *alto)
                        md=np.array([mdx,mdy]) # Este se utiliza para los calculos de angulo
                        #========================

                        #Puntos de talon derecho[tald]
                        taldx=int(marca[28].x *ancho)
                        taldy=int(marca[28].y *alto)
                        tald=np.array([taldx,taldy]) # Este se utiliza para los calculos de angulo
                    
                        #Puntos de talon izquierdo[tali]
                        talix=int(marca[27].x *ancho)
                        taliy=int(marca[27].y *alto)
                        tali=np.array([talix,taliy]) # Este se utiliza para los calculos de angulo

                        #Puntos de puntapie derecho[tald]
                        pudx=int(marca[32].x *ancho)
                        pudy=int(marca[32].y *alto)
                        pud=np.array([pudx,pudy]) # Este se utiliza para los calculos de angulo
                    
                        #Puntos de puntapie izquierdo[tali]
                        puix=int(marca[31].x *ancho)
                        puiy=int(marca[31].y *alto)
                        pui=np.array([puix,puiy]) # Este se utiliza para los calculos de angulo


                        #Puntos de hombro derecho[hd]
                        hdx=int(marca[12].x *ancho)
                        hdy=int(marca[12].y *alto)
                        hd=np.array([hdx,hdy]) # Este se utiliza para los calculos de angulo

                        #Puntos de hombro izquierdo[hi]
                        hix=int(marca[11].x *ancho)
                        hiy=int(marca[11].y *alto)
                        hi=np.array([hix,hiy]) # Este se utiliza para los calculos de angulo

                        #puntos de cadera derecha[cad]
                        cadx=int(marca[24].x *ancho)
                        cady=int(marca[24].y *alto)
                        cad=np.array([cadx,cady]) # Este se utiliza para los calculos de angulo

                        #puntos de cadera izquiera[cai]
                        caix=int(marca[23].x *ancho)
                        caiy=int(marca[23].y *alto)
                        cai=np.array([caix,caiy]) # Este se utiliza para los calculos de angulo

                        #puntos de rodilla derecha[rd]
                        rdx=int(marca[26].x *ancho)
                        rdy=int(marca[26].y *alto)
                        rd=np.array([rdx,rdy]) # Este se utiliza para los calculos de angulo

                        #puntos de rodilla derecha[rd]
                        rix=int(marca[25].x *ancho)
                        riy=int(marca[25].y *alto)
                        ri=np.array([rix,riy]) # Este se utiliza para los calculos de angulo

                        #puntos de tobillo derecho[td]
                        tdx=int(marca[28].x *ancho)
                        tdy=int(marca[28].y *alto)
                        td=np.array([tdx,tdy]) # Este se utiliza para los calculos de angulo

                        #puntos de tobillo izquierdo[ti]
                        tix=int(marca[27].x *ancho)
                        tiy=int(marca[27].y *alto)
                        ti=np.array([tix,tiy]) # Este se utiliza para los calculos de angulo
                        global ang_cad,ang_rd,ang_hd,ang_cd,ang_cai,ang_ri,ang_esp
                        ang_cad=calcular_angulo(hd,cad,rd) #codo
                        ang_rd=calcular_angulo(cad,rd,td) #rodilla
                        ang_hd=calcular_angulo(cd,hd,cad) #hombro
                        ang_cd=calcular_angulo(md,cd,hd) #cadera
                        ang_cai=calcular_angulo(hi,cai,ri)
                        ang_ri=calcular_angulo(cai,ri,ti)
                        ang_esp=calcular_angulo(hd,cad,im)

                        cv2.line(frame,(cadx,cady),(hdx,hdy),(0,255,0),3)
                        cv2.line(frame,(tdx,tdy),(rdx,rdy),(0,255,0),3) 
                        cv2.line(frame,(rdx,rdy),(cadx,cady),(0,255,0),3)
                        limMin=0
                        limMax=360
                        ElimMin=30
                        ElimMax=60
                        deadlift(ElimMin, ElimMax)
                        
                        if limMin < ang_cd < limMax:
                            cv2.line(frame,(cdx,cdy),(mdx,mdy),(0,255,0),3) 
                            cv2.line(frame,(cdx,cdy),(hdx,hdy),(0,255,0),3)  
                        else:           
                            cv2.line(frame,(cdx,cdy),(mdx,mdy),(0,0,255),3)
                            cv2.line(frame,(cdx,cdy),(hdx,hdy),(0,0,255),3) 
                        
                        video_dead_view.update()
                        _,im_arr=cv2.imencode('.png',frame)
                        im_b64= base64.b64encode(im_arr)
                        self.img.src_base64=im_b64.decode('utf-8')
                        self.update()
        
        def build(self):
            self.img=Image(width=400,height=300,
                border_radius=border_radius.all(20)
                )
            
            return Column([
                self.img,
                Row([
                    Text('funciono',size=30,weight='bold',
                         color='white')
                ])
            ])

   
    video_90_view=Container(
        margin=margin.only(bottom=40),
        width=400,
        height=750,
        bgcolor='#6B1A14',
        border_radius=35,
        

        content=Column(
            controls=[
                
                Container(on_click=lambda _:page.go('/'),
                          padding=padding.only(top=20,left=20),
                          
                          content=Text(value='x',size=20)
                          
                          ), 
                Container(
                Text(value='________________________________________________________',size=16.3),
                
                padding=padding.only(top=-1,left=2)),
                Row(alignment='spaceBetween',
                    
                    
                    controls=[
                        Text(value=' ',size=30),
                        Text(
                            value='Sentadilla de 90°',size=38,italic=True
                        ),
                        
                        Row(
                            controls=[
                                Text(value=' ',size=30)
                            ]
                        )
                    ]
                ),              
                
                Container(
                Text(value='________________________________________________________',size=16.3),
                padding=padding.only(left=2)),
                Container(height=20),
                Container(imagen_90()),
                Row(controls=[
                Container(on_click=lambda e:play(e),
                          padding=padding.only(top=60,left=10),
                          
                          content=Icon(
                            icons.PLAY_CIRCLE_OUTLINED,size=75)
                          
                          ),
                Container(on_click=lambda e:page.go('/'),
                          padding=padding.only(top=60,left=75),
                          
                          content=Icon(
                            icons.PAUSE_CIRCLE_OUTLINED,size=70)
                          
                          ),
                Container(on_click=lambda e:stop(e),
                          padding=padding.only(top=60,left=70),
                          
                          content=Icon(
                            icons.STOP_CIRCLE_OUTLINED,size=70)
                          
                          )
                ]
                )
            ]
        )
    )
    
    video_deep_view=Container(
        margin=margin.only(bottom=40),
        width=400,
        height=750,
        bgcolor='#6B1A14',
        border_radius=35,
        

        content=Column(
            controls=[
                
                Container(on_click=lambda _:page.go('/'),
                          padding=padding.only(top=20,left=20),
                          
                          content=Text(value='x',size=20)
                          
                          ), 
                Container(
                Text(value='________________________________________________________',size=16.3),
                
                padding=padding.only(top=-1,left=2)),
                Row(alignment='spaceBetween',
                    
                    
                    controls=[
                        Text(value=' ',size=30),
                        Text(
                            value='Sentadilla profunda',size=38,italic=True
                        ),
                        
                        Row(
                            controls=[
                                Text(value=' ',size=30)
                            ]
                        )
                    ]
                ),              
                
                Container(
                Text(value='________________________________________________________',size=16.3),
                padding=padding.only(left=2)),
                Container(height=20),
                Container(imagen_depp()),
                Row(controls=[
                Container(on_click=lambda _:page.go('/r'),
                          padding=padding.only(top=60,left=10),
                          
                          content=Icon(
                            icons.PLAY_CIRCLE_OUTLINED,size=75)
                          
                          ),
                Container(on_click=lambda _:page.go('/r'),
                          padding=padding.only(top=60,left=75),
                          
                          content=Icon(
                            icons.PAUSE_CIRCLE_OUTLINED,size=70)
                          
                          ),
                Container(on_click=lambda _:page.go('/'),
                          padding=padding.only(top=60,left=70),
                          
                          content=Icon(
                            icons.STOP_CIRCLE_OUTLINED,size=70)
                          
                          )]
                )
            ]
        )
    )

    video_dead_view=Container(
        margin=margin.only(bottom=40),
        width=400,
        height=750,
        bgcolor='#6B1A14',
        border_radius=35,
        

        content=Column(
            controls=[
                
                Container(on_click=lambda _:page.go('/'),
                          padding=padding.only(top=20,left=20),
                          
                          content=Text(value='x',size=20)
                          
                          ), 
                Container(
                Text(value='________________________________________________________',size=16.3),
                
                padding=padding.only(top=-1,left=2)),
                Row(alignment='spaceBetween',
                    
                    
                    controls=[
                        Text(value=' ',size=30),
                        Text(
                            value='Peso muerto',size=38,italic=True
                        ),
                        
                        Row(
                            controls=[
                                Text(value=' ',size=30)
                            ]
                        )
                    ]
                ),              
                
                Container(
                Text(value='________________________________________________________',size=16.3),
                padding=padding.only(left=2)),
                Container(height=20),
                Container(imagen_dead()),
                Row(controls=[
                Container(on_click=lambda _:page.go('/r'),
                          padding=padding.only(top=60,left=10),
                          
                          content=Icon(
                            icons.PLAY_CIRCLE_OUTLINED,size=75)
                          
                          ),
                Container(on_click=lambda _:page.go('/r'),
                          padding=padding.only(top=60,left=75),
                          
                          content=Icon(
                            icons.PAUSE_CIRCLE_OUTLINED,size=70)
                          
                          ),
                Container(on_click=lambda _:page.go('/'),
                          padding=padding.only(top=60,left=70),
                          
                          content=Icon(
                            icons.STOP_CIRCLE_OUTLINED,size=70)
                          
                          )]
                )
            ]
        )
    )
    
    tasks= Column()


    first_page_contents=Container(
        content=Column(
            controls=[
                Text(value='________________________________________________________',size=14.8),
                Row(alignment='spaceBetween',
                    
                    controls=[
                        Container(on_click=lambda e: shrink(e),
                            content=Icon(
                            icons.MENU,size=30)
                        ),
                        Text(
                            value='Intelligent Sport',size=38,italic=True
                        ),
                        
                        Row(
                            controls=[
                                Icon(icons.HISTORY_TOGGLE_OFF_ROUNDED,size=30)
                            ]
                        )
                    ]
                ),
                #Container(height=20),
                Text(value='________________________________________________________',size=14.8),
                #Text(value='Ejercicios'),
                
                Container(height=10),
                Stack(
                    controls=[
                        tasks,
                        FloatingActionButton(
                            text="Sentadilla 90°",width=350,height=150,bgcolor='#0F0000',on_click=lambda _: page.go('/video_90°')                   
                            
                        )
                        
                    ]
                ),
                Container(height=10),
                Stack(
                    controls=[
                        tasks,
                        FloatingActionButton(
                            "Sentadilla profunda",width=350,height=150,bgcolor='#0F0000',on_click=lambda _: page.go('/video_profunda')
                        )
                        
                    ]
                ),
                Container(height=10),
                Stack(
                    controls=[
                        tasks,
                        FloatingActionButton(
                            "Peso muerto",width=350,height=150,bgcolor='#0F0000',on_click=lambda _: page.go('/video_muerto')
                            
                            
                        )
                        
                    ]
                ),
                
                
                
            ],
        ),
    )
    page_1=Container( #pagina de ayuda
        width=400,
        height=750,
        bgcolor='#4F130F',
        border_radius=35,
        

        content=Column(
            controls=[
                
                Container(
                Text(value='________________________________________________________',size=14.8),
                
                padding=padding.only(top=40,left=20)),
                Row(alignment='spaceBetween',
                    
                    
                    controls=[
                        Container(on_click=lambda e: restore(e),
                          content=Icon(
                            icons.MENU,size=30,rotate=1.59)),
                        Text(
                            value='Ayuda',size=38,italic=True
                        ),
                        
                        Row(
                            controls=[
                                Text(value=' ',size=30)
                            ]
                        )
                    ]
                ),              
                
                Container(
                Text(value='________________________________________________________',size=14.8),
                padding=padding.only(left=20)),
                Container(height=10),
                Row( controls=[Container(padding=padding.only(left=10)),
                Stack(
                    controls=[
                        tasks,
                        FloatingActionButton(
                            text="Sentadilla 90°",width=350,height=150,bgcolor='#0F0000',on_click=lambda _: page.go('/video_90°_help')              
                            
                        ),
                        
                    ]
                )]),
                Container(height=10),
                Row( controls=[Container(padding=padding.only(left=10)),
                Stack(
                    controls=[
                        tasks,
                        FloatingActionButton(
                            "Sentadilla profunda",width=350,height=150,bgcolor='#0F0000',on_click=lambda _: page.go('/video_profunda_help')
                        )
                        
                    ]
                )]),
                Container(height=10),
                Row( controls=[Container(padding=padding.only(left=10)),
                Stack(
                    controls=[
                        Container(padding=padding.only(left=20)),
                           
                        
                        tasks,
                        FloatingActionButton(
                            "Peso muerto",width=350,height=150,bgcolor='#0F0000',on_click=lambda _: page.go('/video_muerto_help')
                            
                            
                        )
                        
                    ]
                )])
            ]
        )
    )
    page_2=Row(alignment='end',
        controls=[
            Container(
                width=400,
                height=750,
                bgcolor='#6B1A14',
                border_radius=35,
                animate=animation.Animation(400,AnimationCurve.DECELERATE),
                animate_scale=animation.Animation(10,curve='decelerate'),
                padding=padding.only(
                    top=40,left=20,
                    right=20,bottom=5    
                ),
                content=Column(
                    controls=[
                        first_page_contents
                    ]        
                )        
            )            
        ]
    )
    container = Container(
        width=400,
        height=750,
        bgcolor='#4F130F',
        border_radius=35,
        content=Stack(
            controls=[
                page_1,
                page_2
                
            ]
        )
    )

    pages={
        '/':View(
                "/",
                [
                    container
                ],
            ),
        '/video_90°':View(
                "/video_90°",
                [
                    video_90_view
                ],
            ),
        '/video_profunda':View(
                "/video_profunda",
                [
                    video_deep_view
                ],
            ),
        '/video_muerto':View(
                "/video_muerto",
                [
                    video_dead_view
                ],
            )

    }
   

    def route_video90(route):
        page.views.clear()
        page.views.append(
            pages[page.route]
        )

    
    page.add(container)

    page.on_route_change=route_video90
    page.go(page.route)
    
app(target=main)
