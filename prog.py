-import pandas as pd 
import numpy as np 
from PIL import Image
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.grid_search import GridSearchCV
from skimage.transform import resize
from sklearn.svm import SVC
import pickle

no_of_training_data = 6500
train_data = pd.read_csv('labels.csv')

# getting the id's of the images so that we can loop over 
ids = []
ids = train_data['id'].values


# converting the name of dogs breed into numbers 
label_encoder = LabelEncoder().fit(train_data['breed'])
train_y = label_encoder.transform(train_data['breed'])

y_train = train_y[0:no_of_training_data]



X_train= []

i =0
while i<no_of_training_data:
	img_name = 'train/'+ids[i] + '.jpg'
	im = Image.open(img_name)
	im_grey = im.convert('L') # convert the image to *greyscale*
	img = im_grey.resize((200,200), Image.ANTIALIAS)
	im_array = np.array(img).flatten()
	X_train.append(im_array)
	print('image: ',i)
	i = i+1
'''
X_test = []
i =100
while i<=199:
	img_name = 'train/'+ids[i] + '.jpg'
	im = Image.open(img_name)
	im_grey = im.convert('L') # convert the image to *greyscale*
	img = im_grey.resize((200,200), Image.ANTIALIAS)
	im_array = np.array(img).flatten()
	X_test.append(im_array)
	print('image: ',i)
	i = i+1

'''
# using support vector machines 
print('using support vector machines : ')
Cs = [0.001]
gammas = [0.001]
param_grid = {'C': Cs, 'gamma' : gammas}
model = GridSearchCV(SVC(kernel='rbf', probability =True), param_grid, cv=2)
model.fit(X_train, y_train)
print(model.best_params_)
print(model.grid_scores_)

filename = 'model_4.pickle'
pickle.dump(model, open(filename, 'wb'))

#print(y_test)