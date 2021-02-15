import cv2
import os

img_folder = "./minh_fluorescence/test2/"
save_folder = "./data/test/images2/"
mask_folder = "./minh_fluorescence/mask_image/"

# list = os.listdir(img_folder)
# numfile = len(list)
width    = 640
height   = 480

def resize_imgs(input, output, width, height):
    numfile = 0
    for filename in sorted(os.listdir(input)):
        img = cv2.imread(input+ filename)
        img = cv2.resize(img, (width, height))
        name_save = output + "%d.png"%numfile
        cv2.imwrite(name_save, img)
        numfile+=1

def mask_create(input, output):
    list = os.listdir(input)
    numfile = len(list)
    for i in range(numfile):
        img = cv2.imread(input+ "%d.png"%i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img, thresh = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
        cv2.imwrite(output + "%d.png"%i, thresh)


resize_imgs(img_folder, save_folder, width, height)
# mask_create(save_folder, mask_folder)