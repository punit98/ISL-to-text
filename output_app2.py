import hunspell
import os
import cv2
import numpy as np
import tkinter as tk
import operator
import time
import sys

import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from keras.models import model_from_json
from string import ascii_uppercase


folder = "test/"

class Application:
    def __init__(self):
        os.chdir("Saved Model")
        self.directory = 'Saved Model'

        self.vs = cv2.VideoCapture(0)

        self.current_image = None
        self.current_image = None

        self.json_file = open('Saved_Model.json', 'r')
        self.model_json = self.json_file.read()
        self.json_file.close()
        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights("Saved_Weights.h5")

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0

        for i in ascii_uppercase:
            self.ct[i] = 0
        print ("Model successfully loaded")

        self.root = tk.Tk()
        self.root.title("ISL to Text Conversion")
        self.root.protocol("WM_DELETE_WINDOW", self.destructor)
        self.root.geometry("1350x900")


        self.panel = tk.Label(self.root)
        self.panel.place(x = 135, y = 10, width = 1080, height = 720)

        self.panel2 = tk.Label(self.root)
        self.panel2.place(x = 360, y = 335, width = 600, height = 400)


        self.T = tk.Label(self.root)
        self.T.place(x = 31, y = 17)
        self.T.config(text = "ISL to text Converter", font = ("calibri", 20, "bold"))


        self.panel3 = tk.Label(self.root)
        self.panel3.place(x = 500, y = 640)
        self.T1 = tk.Label(self.root)
        self.T1.place(x = 10, y = 640)
        self.T1.config(text = "Current Character: ", font = ("calibri", 20, "bold"))


        self.panel4 = tk.Label(self.root)
        self.panel4.place(x = 220, y = 700)
        self.T2 = tk.Label(self.root)
        self.T2.place(x = 10, y = 700)
        self.T2.config(text = "This Word: ", font = ("calibri", 20, "bold"))


        self.panel5 = tk.Label(self.root)
        self.panel5.place(x = 350, y = 760)
        self.T3 = tk.Label(self.root)
        self.T3.place(x = 10, y = 760)
        self.T3.config(text = "Whole Sentence: ", font = ("calibri", 20, "bold"))


        self.str = ""
        self.word = ""
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.video_loop()




    


    def video_loop(self):

        ok, screen = self.vs.read()

        if ok:
            captured_image_cv2 = cv2.flip(screen, 1)

            x1 = 0
            y1 = 0
            x2 = 1080
            y2 = 720

            cv2.rectangle(screen, (360, 335), (910, 785), (255, 0, 0), 1)
            captured_image_cv2 = cv2.cvtColor(captured_image_cv2, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(captured_image_cv2)
            image_tk = ImageTk.PhotoImage(image = self.current_image)
            self.panel.image_tk = image_tk
            self.panel.config(image = image_tk)
            captured_image_cv2 = captured_image_cv2[335:935, 360:960]
            Grayscale = cv2.cvtColor(captured_image_cv2, cv2.COLOR_BGR2GRAY)
            Blurred = cv2.GaussianBlur(Grayscale, (9, 9), 5)
            Threshold = cv2.adaptiveThreshold(Blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            res, Res = cv2.threshold(Threshold, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            captured_image_cv2 = cv2.resize(captured_image_cv2, (128, 128))
            self.predic(Res)

            self.current_image2 = Image.fromarray(Res)

            image_tk = ImageTk.PhotoImage(image = self.current_image2)
            self.panel2.image_tk = image_tk
            self.panel2.config(image = image_tk)
            self.panel3.config(text = self.current_symbol, font = ("calibri", 30))
            self.panel4.config(text = self.word, font = ("calibri", 30))
            self.panel5.config(text = self.str, font = ("calibri", 30))


        

        self.root.after(30, self.video_loop)




    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(folder + 'output_test' + '.jpg', Res)

    def predic(self, test):


        test = cv2.resize(test, (128,128))
        res = self.loaded_model.predict(test.reshape(1, 128, 128, 1))
        Predicted = {}
        Predicted['blank'] = res[0][0]
        index = 1


        for i in ascii_uppercase:
            Predicted[i] = res[0][index]
            index += 1



        Predicted = sorted(Predicted.items(), key = operator.itemgetter(1), reverse = True)
        self.current_symbol = Predicted[0][0]


        if(self.current_symbol == 'blank'):

            for i in ascii_uppercase:
                self.ct[i] = 0

        self.ct[self.current_symbol] += 1


        if(self.ct[self.current_symbol] >60):
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue


                temp = self.ct[self.current_symbol] - self.ct[i]


                if temp < 0:
                    temp *= -1


                if temp <= 20:
                    self.ct['blank'] = 0
                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return


            self.ct['blank'] = 0


            for i in ascii_uppercase:
                self.ct[i] = 0

            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1

                    if len(self.str) > 0:
                        self.str += " "

                    self.str += self.word
                    self.word = ""
            else:
                if len(self.str) > 16:
                    self.str = ""

                self.blank_flag = 0
                self.word += self.current_symbol



    def action1(self):
        pred = self.suggest(self.word)

        if len(pred) > 0:
            self.word = ""
            self.str += " "
            self.str += pred[0]

    def action2(self):
        pred = self.suggest(self.word)

        if len(pred) > 1:
            self.word = ""
            self.str += " "
            self.str += pred[1]

    def action3(self):
        pred = self.suggest(self.word)

        if len(pred) > 2:
            self.word = ""
            self.str += " "
            self.str += pred[2]

    def action4(self):
        pred = self.suggest(self.word)

        if len(pred) > 3:
            self.word = ""
            self.str += " "
            self.str += pred[3]

    def action5(self):
        pred = self.suggest(self.word)

        if len(pred) > 4:
            self.word = ""
            self.str += " "
            self.str += pred[4]

    def destructor(self):
        print("Exiting the application")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


    def destructor1(self):
        print("Exiting the application")
        sefl.root1.destroy()

    def action_call(self):
        self.root1 = tk.Toplevel(self.root)
        self.root1.title("About")
        self.root1.protocol('WM_DELETE_WINDOW', self.destructor1)
        self.root1.geometry("900x900")


print ("Starting the program")
fin = Application()
fin.root.mainloop()
