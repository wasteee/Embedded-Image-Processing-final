import cv2
import io
import socket
import struct
import time
import pickle
import zlib
import tkinter as tk
from PIL import Image,ImageTk
import xlrd
import pickle
import numpy as np
import time

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', 8081))
connection = client_socket.makefile('wb')

#Camara
camera = cv2.VideoCapture(0) 
# 建立主視窗和 Frame（把元件變成群組的容器）
window = tk.Tk()
top_frame = tk.Frame(window)
window.geometry("1080x720")
panel = tk.Label(window)  # initialize image panel
panel.pack(padx=10, pady=10, side=tk.LEFT)
window.config(cursor="arrow")
#建立顯示
_0552072=tk.StringVar()
_0552062=tk.StringVar()
_0552050=tk.StringVar()
_Unknow=tk.StringVar()
ShowOut=np.array([False,False,False,False])
# 將元件分為 top/bottom 兩群並加入主視窗
top_frame.pack()
bottom_frame = tk.Frame(window)
bottom_frame.pack(side=tk.BOTTOM)



img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
payload_size = struct.calcsize(">L")
def send2server():
    ret, frame = camera.read()
    result, frame = cv2.imencode('.jpg', frame, encode_param)
    data = pickle.dumps(frame, 0)
    size = len(data)


    print("{}: {}".format(img_counter, size))
    client_socket.sendall(struct.pack(">L", size) + data)
    print('client send')
    #recv
    data = b""
    while len(data) < payload_size:
        print("Recv: {}".format(len(data)))
        data += client_socket.recv(4096)
    print("Done Recv: {}".format(len(data)))
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    print("msg_size: {}".format(msg_size))
    while len(data) < msg_size:
        data += client_socket.recv(4096)
    print('client recv')
    frame_data = data[:msg_size]
    data = data[msg_size:]
    frame_recv=pickle.loads(frame_data, fix_imports=True, encoding="bytes")    
    frame_recv = cv2.imdecode(frame_recv, cv2.IMREAD_COLOR) 
    print('client recv1')
    client_socket.sendall(b"ok")
    result_pre = client_socket.recv(64)
    print('client recv2')
    return frame_recv,result_pre



def video_loop():
   
    
    success, img = camera.read()  # 從攝像頭讀取照片
    
    if success:
        time.sleep(0.001)
        img,result_pre = send2server()
        result_pre = bytes.decode(result_pre)
        print(result_pre)
        if(result_pre[0] == '1'):
            _0552050.set('0552050')
        else:
            _0552050.set('....')
            #----------------------------------
        if(result_pre[1] == '1'):
            _0552072.set('0552072')
        else:
            _0552072.set('....')
            #----------------------------------
        if(result_pre[2] == '1'):
            _0552062.set('0552062')
        else:
            _0552062.set('....')
            #----------------------------------
        if(result_pre[3] == '1'):
            _Unknow.set('Unknow')
        else:
            _Unknow.set('....')
        
        
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)#轉換顏色從BGR到RGBA
        current_image = Image.fromarray(cv2image)#將影象轉換成Image物件
        imgtk = ImageTk.PhotoImage(image=current_image)
        panel.imgtk = imgtk
        panel.config(image=imgtk)
        window.after(1, video_loop)

    
    
# 以下為 top 群組
# 讓系統自動擺放元件，預設為由上而下（靠左）
#left_button = tk.Button(top_frame, text='Close_Camera', fg='black', command=close_video)
#left_button.pack(side=tk.LEFT)
#middle_button = tk.Button(top_frame, text='Excel', fg='green',command=call_Excel)
#middle_button.pack(side=tk.LEFT)
_0552072_lable=tk.Label(window, textvariable=_0552072).place(x=700,y=300)
_0552050_lable=tk.Label(window, textvariable=_0552050).place(x=700,y=350)
_0552062_lable=tk.Label(window, textvariable=_0552062).place(x=700,y=400)
_Unknow_lable=tk.Label(window, textvariable=_Unknow).place(x=700,y=450)
while True:
    video_loop()
    window.mainloop()
    
cv2.waitKey(0)
cv2.destroyAllWindows()










