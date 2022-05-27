import os
import pickle
import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
# to give categories for the dataset
Categories = ['Normal', 'Abnormal']
print("Type y to give categories or type n to go with classification of Normal,Abnormal")
# to read the image
while True:
    check = input()
    if check == 'n' or check == 'y':
        break
    print("Please give a valid input (y/n)")
if check == 'y':
    print("Enter How Many types of Images do you want to classify")
    n = int(input())
    Categories = []
    print(f'please enter {n} names')
    for i in range(n):
        name = input()
        Categories.append(name)
    print(f"")
# to load the image and read it n
flat_data_arr = []
target_arr = []
# please use datadir='/content' if the files are upload on to google collab
# else mount the drive and give path of the parent-folder containing all category images folders.
datadir = 'D:\\dataset'
for i in Categories:
    print(f'loading... category : {i}')
    path = os.path.join(datadir, i)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (150, 150, 3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
df = pd.DataFrame(flat_data)
df['Target'] = target
# split of data
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)
print('Splitted Successfully')
# svm
svm=SVC(probability=True, random_state=2)
svm.fit(x_train, y_train)
y_perd= svm.predict(x_test)
# to save the trained model
svm_class="finalmodelsvm.sav"
pickle.dump(svm, open(svm_class, 'wb'))
# to load the saved model
loaded_model= pickle.load(open(svm_class, 'rb'))
# to print the accuracy od the model
accuracy= accuracy_score(y_test, y_perd)
print("Model's accuracy is:", accuracy)
# to plot confusion matrix
cf_matrix= confusion_matrix(y_test, y_perd)
pl.matshow(cf_matrix)
pl.title("The confusion matrix for the svm model")
print(cf_matrix)
sns.heatmap(cf_matrix, annot=True)
# perdiction of the models accuracy
classification_report(y_test, y_perd)
# to get every test and train models matrix
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)
# testing the model
# converting the data into array and changing the row and columns
x= np.array(df.loc[156]).reshape(1, 67500)
perdiction= svm.predict(x)
perdiction
# to verify the result
y_test[156]