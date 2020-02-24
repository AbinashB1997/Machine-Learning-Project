from sklearn import datasets  #importing All the variety of Datas from Datasets
#from sklearn.svm import SVC
from sklearn import tree
#from sklearn.ensemble import RandomForestClassifier
from scipy import misc
import matplotlib.pyplot as plt
#training part...

digits = datasets.load_digits()
features, labels = digits.data,digits.target


#testing part...

#clf = SVC(gamma = 0.0001, C=100)
clf = tree.DecisionTreeClassifier()
#clf = RandomForestClassifier()
clf.fit(features, labels)
img = misc.imread("Images/seven.png")   # Give The Path Of The Image
img = misc.imresize(img, (8,8))
img = img.astype(digits.images.dtype)
img = misc.bytescale(img)

x_test = []

for i in img:
	for j in i:
		x_test.append(sum(j)/3.0)
plt.title("My prediction Is Like This ! ")
plt.imshow(img, cmap = plt.cm.gray_r, interpolation = "nearest")
print("Wanna See My prediction ?[Y/N]")
c = input()
if(c == 'y' or c == 'Y'):
    plt.show()
else:
    print("Ok ! No Problem...\n")
#prediction part...
print("I Think It Is : ", *clf.predict([x_test]), end='')
