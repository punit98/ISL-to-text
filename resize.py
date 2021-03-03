from PIL import Image
import string
import os, sys


#for i in string.ascii_uppercase:

path = "/home/parallels/Desktop/project final working/Dataset2/Test Data/0/" #+ i +"/"
dirs = os.listdir( path )
print(path)
print(len(dirs))

op = "/home/parallels/Desktop/project final working/change/Dataset/Test Data/0/" #+ i + "/"




for item in dirs:
    #print(item)
    isFile = os.path.isfile(os.path.join(path, item))
    #print(isFile)
    if isFile:
        
        im = Image.open(path+item)
        #op, e = os.path.splitext(path+item)
        imResize = im.resize((128,128), Image.ANTIALIAS)
        imResize.save(op + item, 'JPEG', quality=90)
        im.close()
        #print("Resized")


