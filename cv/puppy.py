import cv2

img = cv2.imread('data/00-puppy.jpg')

while True:
    cv2.imshow('Puppy', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
