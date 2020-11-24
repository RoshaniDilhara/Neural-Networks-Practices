#!/usr/bin/env python
# coding: utf-8

# In[34]:


import tkinter as tk
import numpy as np
from PIL import ImageTk,Image,ImageDraw
import cv2

# import h5 file of trained data
# we are not gonna train the NN here.Same NN architecture is defined here
from keras.models import Sequential
from keras.layers import Dense,Flatten

model = Sequential() #creating a new sequaentail type empty NN model

#we cant directly apply the 2D image to te NN -> so have to flatten the image-> 1x (28x28)
#convert 2D image to flatten layer
#784 neurons for the first layer
#this is the first inpiut layer
model.add(Flatten(input_shape=(28,28)))

#this is a classification problem 
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax')) #here we have 10 types of numbers: 0-9

#compile the model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#without training we can directly load the weight file
model.load_weights('FFNN-MNIST.h5')


#defining event_function. it is a customized function
def event_function(event):
    #draw according to the button motion
    x=event.x
    y=event.y
    
    x1=x-30
    y1=y-30
    
    x2=x+30
    y2=y+30
    
    #creating an oval
    canvas.create_oval((x1,y1,x2,y2),fill='black')
    #when something is drawn in the canvas, at the same time similar image is drawn
    #pillow does not support oval
    img_draw.ellipse((x1,y1,x2,y2),fill='white') # this is passed to the NN
    
# can use this even to create a dataset
def save():
    global count
    #when the save button is pressed, whtever drawn in the canvas is taken into the numpy array
    img_array = np.array(img)
    #resize using a function in openCV
    img_array = cv2.resize(img_array,(28,28))
    #save
    cv2.imwrite(str(count)+'.jpg',img_array)
    count = count+1
    
# when the clear button is called, have to clear both canvas object and the image object
def clear():
    global img,img_draw
    canvas.delete('all')
    img = Image.new('RGB',(500,500),(0,0,0))
    img_draw = ImageDraw.Draw(img)
    
#take the image and apply to NN
def predict():
    #image should be in the same format as the training images
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(img_array,(28,28))
    img_array = img_array/255.0
    img_array = img_array.reshape(1,28,28)
    result = model.predict(img_array)
    label = np.argmax(result,axis=1)
    
    label_status.config(text='PREDICTED' + str(label))
    
count = 0

# Creating the main window and adding widgets to it

window = tk.Tk()
# run the main window(still empty)
# window.mainloop()

# Adding widgets to the empty window

#adding a canvas widget
#parameter 1- what the window this canva should be placed
#then width, height and back ground color + other necessary parameters
canvas = tk.Canvas(window,width = 500,height = 500, bg = 'white')

#place where this canvas should be placed in the main window
# have 3 methods of placing widgets in tkinter windows => grid, place, pack
canvas.grid(row=0,column=0,columnspan=4) #can place 4 objects under this column

# Adding buttons
#whatever the function we are gonna mention in the command keyword will be called when the button is pressed
buttonSave = tk.Button(window,text='SAVE',bg='green',fg='white',font='Helvetica 20 bold',command=save)
# placing the button
buttonSave.grid(row=1, column = 0)

buttonPredict = tk.Button(window,text='PREDICT',bg='blue',fg='white',font='Helvetica 20 bold',command=predict)
buttonPredict.grid(row=1, column = 1)

buttonClear = tk.Button(window,text='CLEAR',bg='yellow',fg='white',font='Helvetica 20 bold',command=clear)
buttonClear.grid(row=1, column = 2)

buttonExit = tk.Button(window,text='EXIT',bg='red',fg='white',font='Helvetica 20 bold',command=window.destroy)
buttonExit.grid(row=1, column = 3)

# adding a label widget
label_status = tk.Label(window,text='PREDICTED DIGIT: NONE',bg='white',font='Helvetica 24 bold')
label_status.grid(row=2,column=0,columnspan=4)

# writing the functionality
# binding the canvas with '<B1-Motion>' bind function
# if you are gonna move the button 1 inside canvas, it is gonna call event_function
canvas.bind('<B1-Motion>',event_function)


#we cant get this canvas into a numpy arrray. We will create another object that can be taken to a numpy array.
#User sees the canvas. Similary we are gonna draw this parallely in another object as well. That object will be passes to the convolutional NN
#to do that, need to get the pillow library
# create an image
#bakcgrnd should be black and foregrnd should be white bcz testing data should be similar to the trained data
img = Image.new('RGB',(500,500),(0,0,0))

# create a drawing object-this is the object that we are gonna draw the image
img_draw = ImageDraw.Draw(img)

# run the main window
window.mainloop()


# In[ ]:





# In[ ]:




