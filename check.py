import cv2
import numpy as np


img = cv2.imread('samples/train-detector/00011.jpg')
ano = 'samples/train-detector/00011.txt'
with open(ano, 'r') as f:
    c = f.readlines()[0].strip()
    c = c.split(',')
    pts = np.array([float(x) for x in c[1:1+2*int(c[0])]])
    tmp = pts.reshape(2, int(c[0]))
    print(tmp)
    print(tmp.T.reshape(-1, 2))
#     pts = pts.reshape(-1, 2)
    pts = tmp.T.reshape(-1, 2)


pts = pts * np.array([img.shape[1], img.shape[0]])
print(img.shape[:2])

pts = pts.reshape(-1, 1, 2).astype(int)
print(pts)
cv2.polylines(img, [pts], True, (0, 255, 0), 3)
cv2.imwrite('check.jpg', img)