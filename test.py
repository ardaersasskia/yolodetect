# import os
# filename='usb_x10y2.5.jpg'
# # 去除.jpg
# base_name=os.path.splitext(filename)[0]
# # 去除usb_
# x_y_=base_name.split('_')[1]
# # 取得x,y
# x_=float(x_y_.split('y')[0].split('x')[1])
# y_=float(x_y_.split('y')[1])
# print(x_,y_)
import os
import cv2
directory = 'dataset/testpic'
for filename in os.listdir(directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path)
        
        # base_name=os.path.splitext(filename)[0]
        # # 去除usb_
        # x_y_=base_name.split('_')[1]
        # # 取得x,y
        # x_=float(x_y_.split('y')[0].split('x')[1])
        # y_=float(x_y_.split('y')[1])
        print(f"{filename} loaded")  
        cv2.imshow('image', img)    
        cv2.waitKey(0)    
import pandas as pd 
df=pd.read_excel('dataset/testpic/0417truexyz.xlsx')
x_=df['x+0.8(cm)'].to_numpy()
y_=df['y(cm)'].to_numpy()
z_=df['z+1.7(cm)'].to_numpy()
print(x_)
print(y_)
print(z_)