import cv2
import imutils

cascade_src = r'E:\New folder\AI Projects\Vehicle Detection\cars.xml'
car_cascade = cv2.CascadeClassifier(cascade_src)
cam = cv2.VideoCapture(0)
cam1 = cv2.VideoCapture(1)

while True:
  detected = 0
  _,img = cam.read() #reading frame
  _,img1 = cam.read()
  
  img = imutils.resize( img, width = 500)
  img1 = imutils.resize( img1, width = 500)
  
  gray =  cv2.cvtColor( img, cv2.COLOR_BGR2GRAY) #colour to grayscale
  gray1 =  cv2.cvtColor( img1, cv2.COLOR_BGR2GRAY)
  
  cars = car_cascade.detectMultiScale(gray,1.1,1) #Coordinates of vehicles
  cars1 = car_cascade.detectMultiScale(gray1,1.1,1)


#for north cam
  for (x,y,w,h) in cars:
    cv2.rectangle(img,(x,y),(x+w, y+h),(0,0,255),2)
  cv2.imshow("Frame", img)
  b = str(len(cars))
  a = int(b)
  detected = a
  n = detected

  print( "---------------------------------------------")
  print("North: %d "%(n))
  if( n>=2):
    print( "North More Traffic")
  else:
    print("No Traffic")

    #for south cam
  for (x2,y2,w2,h2) in cars1:
    cv2.rectangle(img1,(x2,y2),(x2+w2, y2+h2),(0,0,200),2)
  cv2.imshow("South Frame", img1)
  b2 = str(len(cars1))
  a2 = int(b2)
  detected = a2
  n2 = detected

  print( "---------------------------------------------")
  print("South: %d "%(n2))
  if( n2>=2):
    print( "South More Traffic")
  else:
    print("No Traffic")

  if cv2.waitKey(33) == 27:
    break


cam.release()
cam1.release()

cv2.destroyAllWindows()
