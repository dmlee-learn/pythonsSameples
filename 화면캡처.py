import numpy as np
from PIL import ImageGrab
import math
import cv2
import cv2 as cv
import time
import numpy as np
from tkinter import ttk

threshold1 = 1
threshold2 = 0
threshold3 = 1
threshold4 = 0


# Region of Interest : 관심영역을 설정하는 함수
def roi(img, vertices):
    # img 크기만큼의 영행렬을 mask 변수에 저장하고
    mask = np.zeros_like(img)
 
    # vertices 영역만큼의 Polygon 형상에만 255의 값을 넣습니다
    cv2.fillPoly(mask, vertices, 255)
 
    # img와 mask 변수를 and (비트연산) 해서 나온 값들을 masked에 넣고 반환합니다
    masked = cv2.bitwise_and(img, mask)
    return masked
 
 
 
# 이미지에 Canny() 함수를 사용해 윤곽선을 따는 함수
def process_img(original_image, fstr):
    
    
    #회색 이미지 만들기
    img_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # 원하는 영역을 만들고
    vertices =  np.array([[20,500], [20,300], [300,200],[500,200], [600,300], [600,500]])
    # roi()를 사용해 그 영역만큼 영상을 자릅니다
    img_gray = roi(img_gray, [vertices])

    #img_gray = cv2.cvtColor(processed_img, cv.COLOR_GRAY2BGR)
    
    img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, threshold1, threshold2, ksize=3)
    img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
    img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, threshold3, threshold4, ksize=3)
    img_sobel_y = cv2.convertScaleAbs(img_sobel_y)

    cdstP = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0);
    cdstP = cv2.cvtColor(cdstP, cv.COLOR_GRAY2BGR)
    cdstP = cv2.putText(cdstP, fstr, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    return cdstP


#마우스 이벤트 남겨둘 화면 구역 선택
def mouse_callback(event, x, y, flags, param):
    cv.putText(cdstP, str(threshold1), (0, 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0))
    
varBreak = True;
tempText = "text";
########### 추가 ##################
prevTime = 0 #이전 시간을 저장할 변수
###################################
# 무한루프를 돌면서
while(varBreak):
    # (0,40)부터 (800,600)좌표까지 창을 만들어서 데이터를 저장하고 screen 변수에 저장합니다
    screen = np.array(ImageGrab.grab(bbox=(0,40,800,800)))
 
    

    ########### 추가 ##################
    #현재 시간 가져오기 (초단위로 가져옴)
    curTime = time.time()

    #현재 시간에서 이전 시간을 빼면?
    #한번 돌아온 시간!!
    sec = curTime - prevTime
    #이전 시간을 현재시간으로 다시 저장시킴
    prevTime = curTime

    # 프레임 계산 한바퀴 돌아온 시간을 1초로 나누면 된다.
    # 1 / time per frame
    fps = 1/(sec)

    # 디버그 메시지로 확인해보기
    #print "Time {0} " . format(sec)
    #print "Estimated fps {0} " . format(fps)

    # 프레임 수를 문자열에 저장
    fstr = "FPS : %0.1f" % fps
    
    # 이미지에 윤곽선만 추출해서 new_screen 변수에 대입합니다
    new_screen = process_img(screen, fstr)
    
    # pygta5-3이라는 이름의 창을 생성하고 이 창에 screen 이미지를 뿌려줍니다
    cv2.imshow('pygta5-4', new_screen)
 
    # 'q'키를 누르면 종료합니다
    if cv2.waitKey(25) & 0xFF == ord('q'):
        varBreak = False;
        cv2.destroyAllWindows();
        break
        q
    if cv2.waitKey(25) & 0xFF == ord('u'):
        threshold1 = threshold1 - 10
        
    if cv2.waitKey(25) & 0xFF == ord('i'):
        threshold1 = threshold1 + 10
    
    if cv2.waitKey(25) & 0xFF == ord('o'):
        threshold2 = threshold2 - 10
        
    if cv2.waitKey(25) & 0xFF == ord('p'):
        threshold2 = threshold2 + 10
    
    if cv2.waitKey(25) & 0xFF == ord('h'):
        threshold3 = threshold3 - 10
        
    if cv2.waitKey(25) & 0xFF == ord('j'):
        threshold3 = threshold3 + 10
    
    if cv2.waitKey(25) & 0xFF == ord('k'):
        threshold4 = threshold4 - 10
        
    if cv2.waitKey(25) & 0xFF == ord('l'):
        threshold4 = threshold4 + 10
