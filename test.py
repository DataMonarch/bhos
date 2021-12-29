import cv2
import numpy as np
import matplotlib.pyplot as plt
#read image

def AspectRatioResize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

img = cv2.imread("Picture1.jpg")

#convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("image window",gray)
kernel_size = 5
# gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

low_thresh = 50
high_thresh = 150
edges = cv2.Canny(gray, low_thresh, high_thresh)

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 30  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 50  # minimum number of pixels making up a line
max_line_gap = 5  # maximum gap in pixels between connectable line segments
line_image = np.copy(img) * 0  # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                    minLineLength=min_line_length, maxLineGap=max_line_gap)

for line in lines:
   x1, y1, x2, y2 = line[0]
   cv2.line(img, (x1, y1), (x2, y2), (0, 0, 128), 1)

# img = AspectRatioResize(img, height=1200)
# edges = AspectRatioResize(edges, height=1200)
%matplotlib qt
fig, [ax, ax1] = plt.subplots(1, 2)
ax.imshow(edges)
ax1.imshow(img)

# for x1,y1,x2,y2 in lines[0]:
#     cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
#
# lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
# cv2.imshow('Image window', img)

# #performing binary thresholding
# kernel_size = 3
# ret,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
#
# #finding contours
# cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#
# #drawing Contours
# radius =2
# color = (30,255,50)
# cv2.drawContours(image, cnts, -1,color , radius)
# # cv2.imshow(image) commented as colab don't support cv2.imshow()
# cv2.imwrite('Image.png', image)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()


def func(kd, rd, re=745, k0=500, rw=0.33):

    skin = (k0/kd - 1)*np.log(rd/rw)
    pir = np.log(re/rw) / (np.log(re/rw) + skin)

    return skin, pir

func_vect = np.vectorize(func)

kd = [100, 20, 300, 200]
rd = [1.6, 0.8, 2, 2.3]

s, pir = func_vect(kd, np.array(rd)+0.33)
s
pir
