import numpy as np
from PIL import ImageGrab
import math
import cv2
import cv2 as cv
import time
import numpy as np
from tkinter import *
from tkinter import ttk

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
    src = original_image
 
    # 원하는 영역을 만들고
    vertices =  np.array([[20,500], [20,300], [300,200],[500,200], [600,300], [600,500]])
    # roi()를 사용해 그 영역만큼 영상을 자릅니다
    src = roi(src, [vertices])
    
    #qsrc = cv2.resize(src, (640, 360))
 
    dst = cv.Canny(src, threshold1, threshold2, None, 3)
 
    #cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    #cdstP = np.copy(cdst)
    '''
    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
 
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(cdst, pt1, pt2, (0, 0, 255), 1, cv.LINE_AA)
 
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
 
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv.LINE_AA)
            cv.putText(cdstP, str(threshold1), (0, 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0))
            cv.putText(cdstP, str(threshold2), (0, 150), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0))
    '''
    cdstP = cv2.putText(dst, fstr, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255))
    
    return cdstP
    
threshold1 = 50
threshold2 = 150
 
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
    if cv2.waitKey(25) & 0xFF == ord('o'):
        threshold1 = threshold1 - 10
        
    if cv2.waitKey(25) & 0xFF == ord('p'):
        threshold1 = threshold1 + 10
    
    if cv2.waitKey(25) & 0xFF == ord('k'):
        threshold2 = threshold2 - 10
        
    if cv2.waitKey(25) & 0xFF == ord('l'):
        threshold2 = threshold2 + 10
