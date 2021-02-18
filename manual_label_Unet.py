import cv2
import os
import numpy as np

input_d = "./data/cell/training/images/"
output_d = "./data/cell/training/manual/"
# list = os.listdir(img_folder)
# numfile = len(list)
width    = 640
height   = 480

xp, yp, i = 0, 0, 0
j = 0
qu = False
click = False
point = []
pointdraw = []
points = np.array([])

def left_click(event, x, y, flags, param):
    global xp, yp, i, j, click, point, pointdraw, points, mask
    if event == cv2.EVENT_RBUTTONDOWN:
        xp, yp = x, y
        click = True
        i += 1

        print("Point:", i, ", Position:", end=" ")
        print([yp, xp], ", BGR value:", end=" ")
        print(img[yp, xp])

        point.append([xp, yp])
        pointdraw.append((xp, yp))
        points = np.array(point)

        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        if (i > 1):
            cv2.line(img, pointdraw[i - 2], pointdraw[i - 1], (0, 0, 255), 1)
        cv2.imshow('image', img)
    elif (event == cv2.EVENT_LBUTTONDBLCLK) & (i >= 3):
        click = False
        cv2.line(img, pointdraw[0], pointdraw[i - 1], (0, 0, 255), 1)
        j += 1
        i = 0
        # print(points)
        print("Acnes number: ", j)
        print("-------------")

        cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.imshow('image', img)
        cv2.imshow('mask', mask)
        # cv2.imwrite("ROI_crop%d.png" % (j), cropped)
        # cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite("ROI_gray_crop%d.png" % (j), cropped_gray)

        point = []
        pointdraw = []
        points = np.array([])


def open_image(intput):
    img = cv2.imread(intput)
    img = cv2.resize(img, (378, 504))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def mask_create(input, output):
    global img, mask
    for path, subdirs, files in os.walk(input):
        for i in range(len(files)):
            print("Image: " + files[i])
            img = cv2.imread(input + files[i])
            mask = np.zeros(img.shape[0:2], np.uint8)
            while(1):
                cv2.imshow('image', img)
                cv2.imshow('mask', mask)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    qu = False
                    break
                if cv2.waitKey(20) & 0xFF == 27:
                    qu = True
                    break
            if qu == True:
                break
            cv2.imwrite(output + files[i], mask)
        if qu == True:
            cv2.destroyAllWindows()
            break


cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)
cv2.setMouseCallback('image', left_click)
cv2.namedWindow('mask')

mask_create(input_d, output_d)


#
# while(1):
#     if not click:
#         while(1):
#             cv2.imshow('image', img)
#             cv2.imshow('mask', mask)
#             if click == True:
#                 click = False
#                 break
#             if cv2.waitKey(20) & 0xFF == 27:
#                 savefile = "%d.png" % (j)
#                 cv2.imwrite(savefile, mask)
#                 click = qu = True
#                 break
#     if cv2.waitKey(20) & 0xFF == 27:
#         break


