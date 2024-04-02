import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)

    cv2.imshow('Current Image',frame)

    if cv2.waitKey(1) & 0xFF == ord('y'):
        name = input("Enter Name: ")
        cv2.imwrite(name + ".jpg",frame)
        cv2.destroyAllWindows()
        break
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()