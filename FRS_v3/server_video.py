import socket
import cv2
import pickle
import struct
import dlib
import numpy as np
from keras.models import load_model
HOST='localhost'
PORT=8081

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST,PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn,addr=s.accept()


payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

#face det
detector = dlib.get_frontal_face_detector()
ShowOut=np.array([False,False,False,False])
#load CNN model
filename = "./cnn_v18.sav"
model = pickle.load(open(filename,'rb'))

while True:
    data = b""
    result_pre = b""
    while len(data) < payload_size:
        print("Recv: {}".format(len(data)))
        data += conn.recv(4096)
    print("Done Recv: {}".format(len(data)))
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    print("msg_size: {}".format(msg_size))
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")    
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)  
    print('server recv')
    
    ## Image processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    dets = detector(gray, 1)
    count=1  
    ShowOut[0], ShowOut[1], ShowOut[2], ShowOut[3] = False, False, False, False
    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        cv2.rectangle(frame, (x2, x1), (y2, y1), (0, 255, 0), 2)
        face = gray[x1:y1,x2:y2]
        face = cv2.resize(face, (64,64))
        X = [] 
        y = []
        training_data = []
        count += 1
        # Face rec
                 
        training_data.append([face, 1])
        for features,label in training_data:
            X.append(features)
            y.append(label)
            X = np.array(X).reshape(-1, 64, 64, 1)
                
            # predict test picture
            predictions = model.predict_classes(X)
      
            if (predictions == 0):
                cv2.putText(frame, "0552050", (x2,x1-15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1, cv2.LINE_AA)
                ShowOut[0]=True 
                #cv2.imwrite("D:\\python\\Face_ID\\"+str(count)+'.jpg', face)  
                #count += 1
            elif(predictions == 1):
                cv2.putText(frame, "0552072", (x2,x1-15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1, cv2.LINE_AA)
                ShowOut[1]=True
                #cv2.imwrite("D:\\python\\Face_ID\\"+str(count)+'.jpg', face)  
                count += 1
            elif(predictions == 2):
                cv2.putText(frame, "0552062", (x2,x1-15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1, cv2.LINE_AA)
                ShowOut[2]=True
                #cv2.imwrite("D:\\python\\Face_ID\\"+str(count)+'.jpg', face)   
                #count += 1
            else:
                cv2.putText(frame, "unknown", (x2,x1-15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1, cv2.LINE_AA)
                ShowOut[3]=True


    for p in range(0,4):
        if(ShowOut[p]):
            result_pre += b"1"
        else:
            result_pre += b"0"
    
    ###Server_Message
    result, frame = cv2.imencode('.jpg', frame, encode_param) 
    data = pickle.dumps(frame, 0) 
    size = len(data)   
    conn.sendall(struct.pack(">L", size) + data)
    print(len(result_pre))
    sendok = conn.recv(2)
    print(sendok)
    conn.sendall(result_pre)
    print('server send2')
    #print(frame.shape)
    #cv2.imshow('ImageWindow',frame)
    cv2.waitKey(1)
