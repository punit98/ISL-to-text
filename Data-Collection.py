import numpy as np
import cv2
import string
import os
import tkinter as tk


if not os.path.exists("Dataset"):
    os.makedirs("Dataset")

if not os.path.exists("Dataset/Training Data"):
    os.makedirs("Dataset/Training Data")

if not os.path.exists("Dataset/Test Data"):
    os.makedirs("Dataset/Test Data")

for i in range(10):
    if not os.path.exists("Dataset/Training Data/" + str(i)):
        os.makedirs("Dataset/Training Data/" + str(i))
    if not os.path.exists("Dataset/Test Data/" + str(i)):
        os.makedirs("Dataset/Test Data/" + str(i))

for i in string.ascii_uppercase:
    if not os.path.exists("Dataset/Training Data/" + i):
        os.makedirs("Dataset/Training Data/" + i)
    if not os.path.exists("Dataset/Test Data/" + i):
        os.makedirs("Dataset/Test Data/" + i)



DataFor = 'Test Data'
folder = 'Dataset/' + DataFor + '/'

minVal = 70

capture = cv2.VideoCapture(0)
interrupt = -1

while True:
    _, img = capture.read()
    img = cv2.flip(img, 1)


    count = {
            'zero': len(os.listdir(folder+"/0")),
            'one': len(os.listdir(folder+"/1")),
            'two': len(os.listdir(folder+"/2")),
            'three': len(os.listdir(folder+"/3")),
            'four': len(os.listdir(folder+"/4")),
            'five': len(os.listdir(folder+"/5")),
            'six': len(os.listdir(folder+"/6")),
            'seven': len(os.listdir(folder+"/7")),
            'eight': len(os.listdir(folder+"/8")),
            'nine': len(os.listdir(folder+"/9")),
            'a': len(os.listdir(folder+"/A")),
            'b': len(os.listdir(folder+"/B")),
            'c': len(os.listdir(folder+"/C")),
            'd': len(os.listdir(folder+"/D")),
            'e': len(os.listdir(folder+"/E")),
            'f': len(os.listdir(folder+"/F")),
            'g': len(os.listdir(folder+"/G")),
            'h': len(os.listdir(folder+"/H")),
            'i': len(os.listdir(folder+"/I")),
            'j': len(os.listdir(folder+"/J")),
            'k': len(os.listdir(folder+"/K")),
            'l': len(os.listdir(folder+"/L")),
            'm': len(os.listdir(folder+"/M")),
            'n': len(os.listdir(folder+"/N")),
            'o': len(os.listdir(folder+"/O")),
            'p': len(os.listdir(folder+"/P")),
            'q': len(os.listdir(folder+"/Q")),
            'r': len(os.listdir(folder+"/R")),
            's': len(os.listdir(folder+"/S")),
            't': len(os.listdir(folder+"/T")),
            'u': len(os.listdir(folder+"/U")),
            'v': len(os.listdir(folder+"/V")),
            'w': len(os.listdir(folder+"/W")),
            'x': len(os.listdir(folder+"/X")),
            'y': len(os.listdir(folder+"/Y")),
            'z': len(os.listdir(folder+"/Z"))
            }

    total = 0
    for i in count:
        total += count[i]

#display text on screen

    cv2.putText(img, "0: " + str(count['zero']), (10,20), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "1: " + str(count['one']), (10,40), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "2: " + str(count['two']), (10,60), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "3: " + str(count['three']), (10,80), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "4: " + str(count['four']), (10,100), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "5: " + str(count['five']), (10,120), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "6: " + str(count['six']), (10,140), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "7: " + str(count['seven']), (10,160), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "8: " + str(count['eight']), (10,180), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "9: " + str(count['nine']), (10,200), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)

    cv2.putText(img, "a: " + str(count['a']), (10,220), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "b: " + str(count['b']), (10,240), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "c: " + str(count['c']), (10,260), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "d: " + str(count['d']), (10,280), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "e: " + str(count['e']), (10,300), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "f: " + str(count['f']), (10,320), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "g: " + str(count['g']), (10,340), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "h: " + str(count['h']), (10,360), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "i: " + str(count['i']), (10,380), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "j: " + str(count['j']), (10,400), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "k: " + str(count['k']), (10,420), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "l: " + str(count['l']), (10,440), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "m: " + str(count['m']), (10,460), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "n: " + str(count['n']), (10,480), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "o: " + str(count['o']), (10,500), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "p: " + str(count['p']), (10,520), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "q: " + str(count['q']), (10,540), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "r: " + str(count['r']), (10,560), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "s: " + str(count['s']), (10,580), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "t: " + str(count['t']), (10,600), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "u: " + str(count['u']), (10,620), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "v: " + str(count['v']), (10,640), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "w: " + str(count['w']), (10,660), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "x: " + str(count['x']), (10,680), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "y: " + str(count['y']), (10,700), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "z: " + str(count['z']), (10,720), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, str(DataFor), (100,20), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)
    cv2.putText(img, "Total : " + str(total), (100,50), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 0), 1)



    #creating ROI


    x1 = int(0.5 * img.shape[1])
    y1 = 10
    x2 = img.shape[1]-10
    y2 = int(0.5 * img.shape[1])

    #bounding box

    #cv2.rectangle(img, (220, 9), (620, 420), (255, 0, 0), 1)

    #RegionOfInterest = img[350:750, 300:850]

    cv2.rectangle(img, (350, 300), (900, 750), (255, 0, 0), 1)

    RegionOfInterest = img[300:750, 350:900]


    cv2.imshow("image", img)

    #Filters applied here

    gray = cv2.cvtColor(RegionOfInterest, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (9,9), 5)

    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    result, RegionOfInterest = cv2.threshold(threshold, minVal, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #display final result after applying filters
    RegionOfInterest = cv2.resize(RegionOfInterest, (550,450))
    cv2.imshow("final_output", RegionOfInterest)

    #start collection process using interrupts

    interrupt = cv2.waitKey(10)

    if interrupt & 0xFF == 27:
        break
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(folder + '0/' + str(count['zero']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(folder + '1/' + str(count['one']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(folder + '2/' + str(count['two']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(folder + '3/' + str(count['three']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('4'):
        cv2.imwrite(folder + '4/' + str(count['four']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('5'):
        cv2.imwrite(folder + '5/' + str(count['five']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('6'):
        cv2.imwrite(folder + '6/' + str(count['six']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('7'):
        cv2.imwrite(folder + '7/' + str(count['seven']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('8'):
        cv2.imwrite(folder + '8/' + str(count['eight']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('9'):
        cv2.imwrite(folder + '9/' + str(count['nine']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(folder + 'A/' + str(count['a']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(folder + 'B/' + str(count['b']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(folder + 'C/' + str(count['c']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(folder + 'D/' + str(count['d']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('e'):
        cv2.imwrite(folder + 'E/' + str(count['e']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('f'):
        cv2.imwrite(folder + 'F/' + str(count['f']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('g'):
        cv2.imwrite(folder + 'G/' + str(count['g']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(folder + 'H/' + str(count['h']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(folder + 'I/' + str(count['i']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('j'):
        cv2.imwrite(folder + 'J/' + str(count['j']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('k'):
        cv2.imwrite(folder + 'K/' + str(count['k']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('l'):
        cv2.imwrite(folder + 'L/' + str(count['l']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('m'):
        cv2.imwrite(folder + 'M/' + str(count['m']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('n'):
        cv2.imwrite(folder + 'N/' + str(count['n']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('o'):
        cv2.imwrite(folder + 'O/' + str(count['o']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('p'):
        cv2.imwrite(folder + 'P/' + str(count['p']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('q'):
        cv2.imwrite(folder + 'Q/' + str(count['q']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('r'):
        cv2.imwrite(folder + 'R/' + str(count['r']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('s'):
        cv2.imwrite(folder + 'S/' + str(count['s']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('t'):
        cv2.imwrite(folder + 'T/' + str(count['t']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('u'):
        cv2.imwrite(folder + 'U/' + str(count['u']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('v'):
        cv2.imwrite(folder + 'V/' + str(count['v']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('w'):
        cv2.imwrite(folder + 'W/' + str(count['w']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('x'):
        cv2.imwrite(folder + 'X/' + str(count['x']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('y'):
        cv2.imwrite(folder + 'Y/' + str(count['y']) + '.jpg', RegionOfInterest)
    if interrupt & 0xFF == ord('z'):
        cv2.imwrite(folder + 'Z/' + str(count['z']) + '.jpg', RegionOfInterest)



   
capture.release()
cv2.destroyAllWindows()

