import cv2 as cv

cap = cv.VideoCapture(0)

while True:
    flag, img = cap.read()
    new_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    new_img = cv.resize(new_img, (560, 480))
    img = cv.resize(img, (560, 480))
    faces = cv.CascadeClassifier("model.xml")

    results = faces.detectMultiScale(new_img, scaleFactor=1.2, minNeighbors=4)

    for (x, y, w, h) in results:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)

    if ord('q') == 0xFF & cv.waitKey(1):
        break
    cv.imshow('Res', img)
