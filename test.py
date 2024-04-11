import os
filename='usb_x10y2.5.jpg'
# 去除.jpg
base_name=os.path.splitext(filename)[0]
# 去除usb_
x_y_=base_name.split('_')[1]
# 取得x,y
x_=float(x_y_.split('y')[0].split('x')[1])
y_=float(x_y_.split('y')[1])
print(x_,y_)
                