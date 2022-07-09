from matplotlib import pyplot as plt
import cv2
import playsound as playsound
import numpy as np
max_val = 10
max_pt = -1
max_kp = 0

orb = cv2.ORB_create()

cap=cv2.VideoCapture(0)
test_img=(cap.read())[1]

hsv = cv2.cvtColor(test_img, cv2.COLOR_BGR2HSV) #convert it to hsv

h, s, v = cv2.split(hsv)
v = cv2.add(v,30)
v[v > 255] = 255
v[v < 0] = 0
final_hsv = cv2.merge((h, s, v))
img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

#test_img = cv2.imread('files/test_100_1.jpg')
#test_img = cv2.imread('files/test_50_2.jpg')
#test_img = cv2.imread('files/test_20_2.jpg')
#test_img = cv2.imread('files/500.jpg')
#test_img = cv2.imread('files/50.jpg')

original = img
cv2.imshow('original', original)

# keypoints and descriptors

(kp1, des1) = orb.detectAndCompute(test_img, None)

training_set = ['files/20.jpg', 'files/50.jpg', 'files/100.jpg', 'files/500.jpg']

for i in range(0, len(training_set)):
	# train image
	train_img = cv2.imread(training_set[i])

	(kp2, des2) = orb.detectAndCompute(train_img, None)

	bf = cv2.BFMatcher()
	all_matches = bf.knnMatch(des1, des2, k=2)

	good = []

	# if good then append to list of good matches
	for (m, n) in all_matches:
		if m.distance < 0.8 * n.distance:
			good.append([m])

	if len(good) > max_val:
		max_val = len(good)
		max_pt = i
		max_kp = kp2

	print(i, ' ', training_set[i], ' ', len(good),'pi')

if max_val != 10:
	print(training_set[max_pt])
	print('good matches ', max_val,'pi')

	train_img = cv2.imread(training_set[max_pt])  
	img3 = cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)
	
	note = str(training_set[max_pt])[6:-4]
	print('\nDetected Note: Rs. ', note)
	if note=='100':
		playsound.playsound("100.mp3",block=True)
	elif note=='20':
		playsound.playsound("20.mp3",block=True)
	elif note=='50':
		playsound.playsound("50.mp3",block=True)
	elif note=='500':
		playsound.playsound("500.mp3",block=True)


	
	(plt.imshow(img3), plt.show())
else:
	print('No Matches')
	(plt.imshow(img),plt.show())
