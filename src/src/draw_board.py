# coding:utf-8
import importlib
import sys
importlib.reload(sys)
import numpy as np 
import cv2
from predict_number import predict

drawing = False  # 当鼠标按下时设置 要进行绘画
mode = False  # 如果mode为True时就画矩形，按下‘m'变为绘制曲线

def drawcircle(event, x, y, flags, param):
    global drawing, mode
    color = (255, 255, 255)

    # 当按下左键时，返回起始的位置坐标
    if event == cv2.cv2.EVENT_LBUTTONDOWN:
        drawing = True
    # 当鼠标左键按下并移动则是绘画圆形，event可以查看移动，flag查看是否按下
    elif event == cv2.cv2.EVENT_MOUSEMOVE and flags == cv2.cv2.EVENT_FLAG_LBUTTON:
        if drawing == True:
            if mode == True:
                cv2.cv2.rectangle(img, (x, y), (x+1, y+1), color, -1)
            else:
                # 绘制圆圈，小圆点连接在一起成为线，1代表了比划的粗细
                cv2.cv2.circle(img, (x, y), 1, color, -1)
        elif event == cv2.cv2.EVENT_LBUTTONUP:
            drawing = False

if __name__ == "__main__":
    img = cv2.cv2.imread('../testdata/src.png')
    cv2.cv2.namedWindow('image', cv2.cv2.WINDOW_NORMAL)
    cv2.cv2.setMouseCallback('image', drawcircle)

    print("Create draw board successfully")
    while True:
        cv2.cv2.imshow('image', img)
        key = cv2.cv2.waitKey(10) & 0xFFF
        if key == ord('m'): # 模式转化
            print("Mode changing..")
            mode = not mode
            print("Done")
        elif key == 27 or key == ord('q'): # 退出程序
            print("Quit")
            break
        elif key == ord('p'): # 进行图像识别
            print("Start to predict the number..")
            save_file = "../testdata/dst.png"
            cv2.cv2.imwrite(save_file, img)
            num = predict(save_file)
            print("Done")
            print("The number is:", num)
            # ui
            cv2.cv2.namedWindow('number', cv2.cv2.WINDOW_NORMAL)
            # number = cv2.cv2.imread('../testdata/src.png')
            number = np.zeros((600, 600, 3), np.uint8)
            cv2.cv2.putText(number, "The number is", (55, 100),
                            cv2.cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255), 4)
            cv2.cv2.putText(number, str(num), (160, 500),
                        cv2.cv2.FONT_HERSHEY_TRIPLEX, 14, (255, 255, 255), 18)
            cv2.cv2.imshow('number', number)
        elif key == ord('c'): # 清空画板，重新写字
            cv2.cv2.destroyWindow('number')

            print("Clearing the board..")
            img = cv2.cv2.imread('../testdata/src.png')
            print("Done")
            
    cv2.cv2.destroyAllWindows()

    