import cv2

# set backend to V4L if not already set
if cv2.CAP_PROP_BACKEND == cv2.CAP_V4L2:
    cv2.CAP_PROP_BACKEND = cv2.CAP_V4L

# open the Logitech C310 webcam
cap = cv2.VideoCapture(2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)


while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret:
        # Display the resulting frame
        cv2.imshow('frame', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything when done
cap.release()
cv2.destroyAllWindows()



