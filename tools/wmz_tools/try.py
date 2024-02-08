import cv2 
image_path = '/mnt/public/usr/wangmingze/Datasets/CD/rs_builds/gt/b_443320.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
cv2.imwrite('/mnt/public/usr/wangmingze/opencd/tools/wmz_tools/try.png', img)
img = cv2.imread('/mnt/public/usr/wangmingze/opencd/tools/wmz_tools/try.png', cv2.IMREAD_UNCHANGED)
import pdb; pdb.set_trace()