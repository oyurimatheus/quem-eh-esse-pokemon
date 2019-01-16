from extract_features import load_training_data#, load_tests
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import cv2

data, labels = load_training_data()

model = LinearSVC()
model.fit(data, labels)


a = model.predict([data[0]])

print(a)
img = cv2.imread('dataset/images/squirtle/squirtle-0.png', 1)

img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img)
print(cv2.moments(img_cinza))
momentos = cv2.moments(img_cinza)

print(cv2.HuMoments(momentos))

#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

poke = ''

if a == 0:
    poke = 'Squirtle'

#plt.title(poke)
#plt.show()