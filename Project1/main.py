# Tkinter modules for widgets and GUI
from tkinter import *
from tkinter import messagebox
from tkinter import ttk  # aditional widgets

# numpy and PIL modules for array functionality and Image proccesing
import numpy as np
from PIL import ImageGrab, Image

# Keras for later usage, MNIST for graphical digits dataset
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Scikit Learn to use ML technics
import sklearn
from sklearn import svm
from sklearn import metrics

# Train and test set sizes. Normally they are 60_000 and 10_000 respectively
size_train_set = 50
size_test_set = 10

# Create raw sets from MNIST data
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = mnist.load_data()

# Show the dimensions of raw data
print(f"Train set x= {x_train_raw.shape} features y= {y_train_raw.shape}")
print(f"Test set x= {x_test_raw.shape} features y= {y_test_raw.shape}")

# check if the raw data dimensions are 28x28 as assumed
if len(x_train_raw[0][1]) == 28 and len(x_train_raw[0][2] == 28):
    # preprocess the data to acceptable dimensions as 2D array
    x_train = x_train_raw.reshape(x_train_raw.shape[0], len(x_train_raw[0][1]) * len(x_train_raw[0][2]))
    # features dimensions do not need to be preprocessed because they are the desired numbers
    y_train = y_train_raw
    x_test = x_test_raw.reshape(x_test_raw.shape[0], len(x_test_raw[0][1]) * len(x_test_raw[0][2]))
    y_test = y_test_raw

    print("Dimensions after slicing arrays:")
    print(f"Train set x={x_train.shape} features y= {y_train.shape}")
    print(f"Test set x={x_test.shape} features y= {y_test.shape}")

    # create classifier estimator instance as SVM model
    clf = svm.SVC()
    # fit to the model using part of the training data (slicing samples number)
    clf.fit(x_train[:size_train_set, ], y_train[:size_train_set, ])

    # make a predictions using sliced test set
    predict = clf.predict(x_test[:size_test_set, ])

    # check accuracy of model using previous predictions and testing features
    acc = metrics.accuracy_score(y_test[:size_test_set, ], predict)
    print(acc)

else:
    print("Data from MNIST dataset is not rigth. Check data dimensions.")


# Creating main window
window = Tk()
window.title("Main window")
window.geometry("1000x750")


### Creating mouse event to draw on canvas


def draw_line(event):
    x, y = event.x, event.y
    r = 14
    if canvas.old_coords:
        x1, y1 = canvas.old_coords
        canvas.create_oval(x - r, y - r, x1 + r, y1 + r, fill="black")
    canvas.old_coords = x, y


def reset_cords(event):
    canvas.old_coords = None


# Canvas instance

canvas = Canvas(window, width=600, height=600, bg="white", cursor="cross")
canvas.pack(anchor="nw")

# Reseting coordinates from canvas
canvas.old_coords = None

# Binding mouse functions to functions
canvas.bind('<B1-Motion>', draw_line)
canvas.bind("<ButtonRelease-1>", reset_cords)

# Making artificial sections on main window
background_lines = Canvas(window, width=400, height=600, bg="grey94")
background_lines.create_line(0, 600, 400, 600, fill="black", width=2)
background_lines.create_line(0, 300, 400, 300, fill="black", width=2)
background_lines.create_line(0, 150, 400, 150, fill="black", width=2)
background_lines.place(x=600, y=0)

### Tkinter GUI functions binded with widgets


def exit_app():  # Close program
    window.destroy()


def number_submit():
    entry_number = entry_item.get()
    print(entry_number)

    if not entry_number.isnumeric():
        submited_number_text = "wrong"
    elif 0 <= int(entry_number) <= 9:
        submited_number_text = f"Added: {str(entry_number)}"
    else:
        submited_number_text = "wrong"

    lbl_submited_number = Label(window,
                                text=submited_number_text,
                                font=("Arial", 20),
                                width=10,
                                height=1)
    lbl_submited_number.place(x=780, y=65)


def clear_canvas():  # Clear canva to blank area
    canvas.delete("all")


def show_statistics():
    lbl_stats_text = Label(window,
                           text="Number predictions:",
                           font=("Arial", 20),
                           width=15,
                           height=1,
                           bg="grey94")
    lbl_stats_text.place(x=730, y=455)


result_number = 0


def submit():
    global result_number
    result_number += 1

    lbl_result = Label(window,
                       text=str(result_number),  # tmp text "result"
                       font=("Arial", 25),
                       width=3,
                       height=1,
                       bg="white")
    lbl_result.place(x=830, y=320)

    # lbl_result_text = Label(window,
    #                         text="Result:",
    #                         font=("Arial", 25),
    #                         width=5,
    #                         height=1,
    #                         bg="grey94")
    # lbl_result_text.place(x=740, y=375)

    # Get the canva corners for Image preprocessing
    x = window.winfo_rootx() + canvas.winfo_x()
    y = window.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    # Create an instance from canva widget to image object
    img = ImageGrab.grab().crop((x, y, x1, y1))  # .show()

    # Pass above instance to function for further processing
    predict_digit(img)


def predict_digit(image):  # Process the data from image canva
    image = image.resize((28, 28))  # resize the image to dataset size
    image = image.convert('L')  # reformat do grayscale
    # image.show()  # show the processed image in new window

    image = np.array(image)  #
    image = image.reshape(1, 784)
    image = image / 255.0
    # print(image)


def choose():
    pass


btn_submit = Button(window,
                    text="submit result",
                    font=("Arial", 25),
                    width=10,
                    height=1,
                    command=submit)
btn_submit.place(x=605, y=310)

btn_choose_algoritm = Button(window,
                             text="choose algorith",
                             font=("Arial", 25),
                             width=14,
                             height=1,
                             command=choose)
btn_choose_algoritm.place(x=605, y=400)

lbl_stats = Label(window,
                  text="Stats",
                  font=("Arial", 25),
                  width=5,
                  height=1)
lbl_stats.place(x=750, y=160)

btn_show_data = Button(window,
                       text="show",
                       font=("Arial", 25),
                       width=5,
                       height=1,
                       command=show_statistics)
btn_show_data.place(x=750, y=220)

lbl_add_record = Label(window,
                       text="Add record",
                       font=("Arial", 23))
lbl_add_record.place(x=720, y=10)

btn_add = Button(window,
                 text="add",
                 font=("Arial", 25),
                 width=5,
                 height=1,
                 command=number_submit)
btn_add.place(x=615, y=50)

entry_item = Entry(window,
                   font=("Arial", 25),
                   width=2)
entry_item.place(x=730, y=65)

btn_exit = Button(window,
                  text="exit",
                  font=("Arial", 25),
                  width=6,
                  height=1,
                  command=exit_app)
btn_exit.place(x=860, y=660)

btn_clear = Button(window,
                   text="clear",
                   font=("Arial", 25),
                   width=6,
                   height=1,
                   command=clear_canvas)
btn_clear.place(x=720, y=660)

window.mainloop()
