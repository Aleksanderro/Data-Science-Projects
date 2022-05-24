#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Aleksander Ratajczyk
# Created Date: 24.05.2022
# email       : ale.ratajczyk@gmail.com
# status      : Prototype, next versions in deveopment
# git         : https://github.com/Aleksanderro/Python_TKinter

# version = 1.0
# ----------------------------------------------------------------------------
""" The main.py file allows to recognise digits handwritten by user on canva and match them with a digit through
a trained models. The procces of training is based on MNIST digits database and classifiers from Scikit-Learn
(Sklearn). The project is supported by GUI from Tkinter modules and can work as individual app conecting Machine
Learning with human interactions. App allows to:
- choose classifiers - models to train,
- number of data samples split into train test data,
- list out statistics to selected and trained models
- add new written digits to a database as csv file "new_records.csv".
Trained models are stored in directory "Saved_models" in working directory. They are dynamically called in function
"classifiers_and_sets_submit()" and used for predictions.
The structure of project is simple - special function "run_once()" and "main()" as main tree branch. "main()" contains
the rest functions.
The app is a first big project which base on ML and user friendly environment. Future versions of project will add more
functionality, better models tuning, intermediate Keras models and will get rid of global variables. """
# ----------------------------------------------------------------------------
# Imports are divided into sections:
# GUI support: Tkinter
# Data operations: keras, MNIST, Sklearn
# Support data visualisation: Matplotlib
# Data transformations and window graber: Numpy, PIL
# Data storing, directory managment, csv operations: pickle, os, csv
# decorator for single function occurence: functools
# ----------------------------------------------------------------------------


import tkinter as tk
from tkinter import messagebox
from tkinter import ttk  # aditional widgets

import keras
from keras.datasets import mnist

import sklearn
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import matplotlib.pyplot as plt

import numpy as np
from PIL import ImageGrab, Image
import PIL.ImageOps as PIL_Ops

import pickle
import os
import csv

from functools import wraps


# run once function using wrapper
def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            result = f(*args, **kwargs)
            wrapper.has_run = True
            return result

    wrapper.has_run = False
    return wrapper


def main():
    """Main function containing the logic of the project."""

    # create raw sets from MNIST data
    (gX_TrainRaw, gY_TrainRaw), (gX_TestRaw, gY_TestRaw) = mnist.load_data()

    # classifiers names list
    gNames = [
        "KNN",
        "SVC",
        "QDA",
        "Decision_Tree",
        "Random_Forest",
        "Neural_Net",
        "AdaBoost",
        "Naive_Bayes"
    ]

    # classifiers with set parameters
    gClassifiers = [
        KNeighborsClassifier(),  # default n_neighbours = 5
        SVC(kernel="linear", C=1.0, cache_size=1500, probability=True),
        # default kernel = "rbf", C = 1.0 - decrease if much noise is present
        QuadraticDiscriminantAnalysis(),
        DecisionTreeClassifier(max_depth=5),  # default max_depth = None
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features="auto"),  # default max_depth = None,
        # n_estimators=100, max_features = "auto"
        MLPClassifier(alpha=1, max_iter=2000),  # default alpha = 0.0001, max_iter=200
        AdaBoostClassifier(),
        GaussianNB()
    ]

    # check sizes of names and classifiers in case of adding or removing elements
    if not len(gNames) == len(gClassifiers):
        print("names and classifiers have diffrent sizes, check it!")
        quit()

    gModelAccs = {key: False for key in gNames}  # list of classifiers and their accuracies
    gSelectedClassifiers = {key: False for key in gNames}  # dict with names and flags set to 0 on init
    gNewRecord = []  # for add_record
    gReports = {}  # report dict

    # working and models directory locations
    gProjectDir = os.getcwd()  # current working directory
    gModelsDir = gProjectDir + "\\Saved_models"
    print(f"Models dir is {gModelsDir}")

    # check models directory existence and possibly create it
    gIsExist = os.path.exists(gModelsDir)
    if not gIsExist:
        os.makedirs(gModelsDir)
        print(f"Creating new directory for models: {gModelsDir}")
    else:
        print(f"Directory already exist! {gModelsDir}")

    # open empty csv file if not exists create one named "new_records.csv"
    with open("new_records.csv", "a", newline='') as empty_csv:  # a -> append
        pass

    # function called by "sets_sizes_submit" in init tab by button
    def classifiers_and_data_sets(train_set_size_, test_set_size_, selected_classifiers_):
        """Check data sizes, create and store models, do models operations, first GUI tab creation."""

        # train and test data sets variables
        train_size = train_set_size_
        test_size = test_set_size_

        # check sets sizes
        if train_size > 60000:
            train_size = 60000
            test_size = test_size

            tk.messagebox.showinfo("Info", "Train set too big, changed to 60_000")
            print("Train set too big, changed to 60_000")

        elif test_size > 10000:
            train_size = train_size
            test_size = 10000

            tk.messagebox.showinfo("Info", "Test set too big, changed to 10_000")
            print("Test set too big, changed to 60_000")

        elif train_size > 60000 and test_size > 10000:
            train_size = 60000
            test_size = 10000

            tk.messagebox.showinfo("Info", "Test and train sets too big, changed to 60_000 and 10_000")
            print("Set too big train and test sets, changed to basic size of 60_000 and 10_000")

        # show the dimensions of raw data
        print(f"Train set x = {gX_TrainRaw.shape}\t features y = {gY_TrainRaw.shape}")
        print(f"Test set x = {gX_TestRaw.shape}\t features y = {gY_TestRaw.shape}\n")

        # check if the raw data dimensions are 28x28 as assumed
        if len(gX_TrainRaw[0][1]) == 28 and len(gX_TestRaw[0][2] == 28):
            # preprocess the data to acceptable dimensions as 2D array of n_samples and k_features (n, 784)
            x_train = gX_TrainRaw.reshape(gX_TrainRaw.shape[0], len(gX_TrainRaw[0][1]) * len(gX_TrainRaw[0][2]))
            x_test = gX_TestRaw.reshape(gX_TestRaw.shape[0], len(gX_TestRaw[0][1]) * len(gX_TestRaw[0][2]))

            # normalize the data
            x_train = x_train / 255.0
            x_test = x_test / 255.0

            # features set
            y_train = gY_TrainRaw
            y_test = gY_TestRaw

            print("Dimensions after slicing arrays:")
            print(f"Train set x = {train_size}, {x_train.shape[1]}\t features y = {train_size}")
            print(f"Test set x = {test_size}, {x_test.shape[1]}\t features y = {test_size}\n")

            print("### - model not used:")
            print()

            # create classifiers estimators instances
            for clf, (key_name, val), (_, val2) in zip(gClassifiers,
                                                       selected_classifiers_.items(),
                                                       gSelectedClassifiers.items()):

                # check if classifier was selected to be initialised
                if val:
                    # fit to the model using part of the training data (slicing samples number)
                    classifier = clf.fit(x_train[:train_size, ], y_train[:train_size, ])

                    # current working directory
                    cwd = os.getcwd()

                    # "Saved_models" path"
                    ms_dir = cwd + "\\Saved_models"

                    # model name and path
                    file_name = key_name + "_model.sav"
                    file_loc = ms_dir + "\\" + file_name

                    # save model with pickle in folder
                    with open(file_loc, "wb") as handler:  # wb -> write, binary
                        pickle.dump(classifier, handler, protocol=pickle.HIGHEST_PROTOCOL)
                        print(f"Saving the model as {file_name}")

                    # load model from folder
                    with open(file_loc, "rb") as handler:  # rb -> read, binary
                        loaded_model = pickle.load(handler)
                        print(f"Loading model {file_name}")

                    # make a predictions using sliced test set
                    predict = clf.predict(x_test[:test_size, ])

                    # check if model of specific algorithm was created, if not - connect clf with its name in dict
                    gSelectedClassifiers[key_name] = loaded_model

                    # check accuracy of model using previous predictions and testing features
                    acc = metrics.accuracy_score(y_test[:test_size, ], predict)
                    gModelAccs[key_name] = acc

                    # fill dictionary with chosen models
                    rep = metrics.classification_report(y_test[:test_size, ], predict, output_dict=True)
                    gReports[key_name] = rep

                    # print accuracy of each classifiers based on given data
                    print(f"{gModelAccs[key_name]} - {key_name}")

                else:
                    print(f"### - {key_name}")

            def top_lvl_destroy():
                """Destroy window after clicking button."""

                initialised_data_win.destroy()

            # ----------------------------------------------------------------------------
            # new window with data printing for user
            initialised_data_win = tk.Toplevel()
            initialised_data_win.title("Proceed window")
            initialised_data_win.geometry("400x250")
            initialised_data_win.configure(background="white")

            ### message box window
            # label 1 "Initialized models:"
            Init_models_lbl = tk.Label(initialised_data_win,
                                       text="Initialized models:",
                                       font=("Arial", 10, 'bold'),
                                       height=1,
                                       bg="white")
            Init_models_lbl.place(x=10, y=5)

            # y position var 1
            y_iter = 30

            # labels showing selected classifiers
            for (key_n, val_n) in gSelectedClassifiers.items():
                if val_n:
                    init_lbl = tk.Label(initialised_data_win,
                                        text=str(key_n),
                                        font=("Arial", 10),
                                        height=1,
                                        anchor="w",
                                        bg="white")
                    init_lbl.place(x=10, y=y_iter)

                    y_iter = y_iter + 20

            # Label 2.1 - train set size
            Init_train_size_lbl = tk.Label(initialised_data_win,
                                           text="Train set = " + str(train_size),
                                           font=("Arial", 10, 'bold'),
                                           height=1,
                                           bg="white")
            Init_train_size_lbl.place(x=200, y=5)

            # Label 2.2 - test set size
            Init_test_size_lbl = tk.Label(initialised_data_win,
                                          text="Train set = " + str(test_size),
                                          font=("Arial", 10, 'bold'),
                                          height=1,
                                          bg="white")
            Init_test_size_lbl.place(x=200, y=25)

            # Label 3 "Models accuracies:"
            Init_accs_lbl = tk.Label(initialised_data_win,
                                     text="Models accuracies: ",
                                     font=("Arial", 10, 'bold'),
                                     height=1,
                                     bg="white")
            Init_accs_lbl.place(x=200, y=55)

            # y position var 2
            y_iter = 75

            # Labels showing models accuracies
            for (key_n, val_n), (_, val_n2) in zip(gSelectedClassifiers.items(), gModelAccs.items()):
                if val_n:
                    Init_models_lbl = tk.Label(initialised_data_win,
                                               text=str(val_n2) + " - " + str(key_n),
                                               font=("Arial", 10),
                                               height=1,
                                               anchor="w",
                                               bg="white")
                    Init_models_lbl.place(x=200, y=y_iter)

                    y_iter = y_iter + 20

            # Button 1 - close info window
            Init_proceed_btn = tk.Button(initialised_data_win,
                                         text="Close",
                                         font=("Arial", 12),
                                         height=1,
                                         command=top_lvl_destroy)
            Init_proceed_btn.pack(anchor="se")

            # ----------------------------------------------------------------------------

        # Wrong data size
        else:
            print("Data from MNIST dataset is not rigth. Check data dimensions.")

    # ----------------------------------------------------------------------------
    window = tk.Tk()  # main window
    window.title("Main window")
    window.geometry("1000x760")

    # mouse event to draw on canvas
    def draw_line(event):
        """Make lines on canva as continuous ovals."""

        x, y = event.x, event.y
        r = 15
        if canvas.old_coords:
            x1, y1 = canvas.old_coords
            canvas.create_oval(x - r, y - r, x1 + r, y1 + r, fill="black")
        canvas.old_coords = x, y

    def reset_cords(event):
        """Delete cordinates on unclicking mouse button."""

        canvas.old_coords = None

    # ----------------------------------------------------------------------------

    def create_selected_models(*args):
        """Select and flag classifiers, set and check data, call 2 functions."""

        # make a dict based on selected classifiers
        selected_classifiers = {obj_: var_.get() for obj_, var_ in classifiers_to_select.items()}

        # flag for default classifier if none is selected
        cls_flag = True

        # loop through selected classifiers
        print(f"Initalized models: ")
        for (key, val) in selected_classifiers.items():
            if val:
                cls_flag = False  # make flag disabled if any classifier was chosen
                print(key)
        print()

        # if any classifiers was not selected make KNN as default selection
        if cls_flag:
            selected_classifiers["KNN"] = True
            classifiers_to_select["KNN"].set(True)
            print("KNN set as default classifier")
            print()

        # make variables for entries values
        entry_train_val = entry_train.get()
        entry_test_val = entry_test.get()

        # check if entries values are numbers
        if entry_train_val.isnumeric() and entry_test_val.isnumeric():
            # make 2500 and 1000 the lowest data sizes
            if int(entry_train_val) < 2500 or int(entry_test_val) < 1000:
                train_set_init = 2500
                entry_train.delete(0, tk.END)
                entry_train.insert(0, train_set_init)

                test_set_init = 1000
                entry_test.delete(0, tk.END)
                entry_test.insert(0, test_set_init)

                tk.messagebox.showinfo("Info", "Values too small, set to max as 60000 and 10000.")

            # if entries are set bigger than 60_000 or 10_000 set them to the max values
            elif int(entry_train_val) > 60000 or int(entry_test_val) > 10000:
                train_set_init = 60000
                entry_train.insert(0, train_set_init)

                test_set_init = 10000
                entry_test.insert(0, test_set_init)

                tk.messagebox.showinfo("Info", "Values too big, set to max as 60000 and 10000.")

            # any other values between 60_000 - 2500 and 10_000 - 1000
            else:
                train_set_init = int(entry_train_val)
                test_set_init = int(entry_test_val)

        # if entries are empty set default values as 2500 and 1000
        elif entry_train_val == "" or entry_test_val == "":

            train_set_init = 2500
            entry_train.delete(0, tk.END)
            entry_train.insert(0, train_set_init)

            test_set_init = 1000
            entry_test.delete(0, tk.END)
            entry_test.insert(0, test_set_init)

            tk.messagebox.showinfo("Info", "Setting default values as 2500 and 1000.")

        # if entries are set to "max" assign them maximum values
        elif entry_train_val == "max" and entry_test_val == "max":
            train_set_init = 60000
            test_set_init = 10000

            tk.messagebox.showinfo("Info", "Setting max values as 60000 and 10000.")

        # wrong entry values, set to default 2500 and 1000
        else:
            train_set_init = 2500
            entry_train.delete(0, tk.END)
            entry_train.insert(0, train_set_init)

            test_set_init = 1000
            entry_test.delete(0, tk.END)
            entry_test.insert(0, test_set_init)

            tk.messagebox.showinfo("Info", "Wrong values entered. Set to 2500 and 1000")

        # disable visually previously selected classifiers
        for chk_btn, (key, val) in zip(check_buttons_list, selected_classifiers.items()):
            if val:
                chk_btn.config(state=tk.DISABLED)

        # call a function to set data sizes and inspect it
        classifiers_and_data_sets(train_set_init, test_set_init, selected_classifiers)

        # call function from operating window to choose only selected models
        choose_model()

    # instance of a tab manager
    notebook = ttk.Notebook(window)

    # tabs: initialization and main operation
    init_tab = tk.Frame(notebook)  # new frame for tab1
    main_tab = tk.Frame(notebook)  # new frame for tab2

    # pack tabs on window
    notebook.add(init_tab, text="Initialize models")
    notebook.add(main_tab, text="Main window")
    notebook.pack(expand=True, fill="both")  # expand to fill any space not otherwise used

    ### init tab (first)
    # label 1.1 - train size
    lbl_entry_train_desc = tk.Label(init_tab,
                                    text="Train data:",
                                    font=("Arial", 10),
                                    bg="grey94")
    lbl_entry_train_desc.place(x=0, y=5)

    # label 1.2 - test size
    lbl_entry_test_desc = tk.Label(init_tab,
                                   text="Test data:",
                                   font=("Arial", 10),
                                   bg="grey94")
    lbl_entry_test_desc.place(x=0, y=30)

    # label 2.1 - train size range
    lbl_entry_range1_init = tk.Label(init_tab,
                                     text="Range: 2500 - 60000",
                                     font=("Arial", 10),
                                     height=1,
                                     bg="grey94")
    lbl_entry_range1_init.place(x=200, y=5)

    # label 2.2 - test size range
    lbl_entry_range2_init = tk.Label(init_tab,
                                     text="Range: 1000 - 60000",
                                     font=("Arial", 10),
                                     height=1,
                                     bg="grey94")
    lbl_entry_range2_init.place(x=200, y=30)

    # entry 1.1 train set
    entry_train = tk.Entry(init_tab)  # train data size
    entry_train.place(x=70, y=5)

    # entry 1.2 train set
    entry_test = tk.Entry(init_tab)  # test data size
    entry_test.place(x=70, y=30)

    # label 3 - QDA info
    lbl_QDA_info = tk.Label(init_tab,
                            text="QDA not recomended with train data over 10000 and test data over 5000",
                            font=("Arial", 10, "italic"),
                            height=1,
                            bg="grey94")
    lbl_QDA_info.place(x=20, y=650)

    # dict - Tkinter variables: clfs names
    classifiers_to_select = {}

    # list to store check buttons
    check_buttons_list = []

    # checkbuttons
    for obj in gNames:
        var = tk.BooleanVar()
        check_button = tk.Checkbutton(init_tab,
                                      text=obj,
                                      variable=var,
                                      onvalue=1,
                                      offvalue=0,
                                      font=("Arial", 24),
                                      fg="black",
                                      bg="grey94",
                                      activeforeground="black",
                                      activebackground="white",
                                      padx=25,
                                      pady=10,
                                      compound="left")

        # store checkbuttons in list for further usage
        check_buttons_list.append(check_button)

        # connect keys with values
        classifiers_to_select[obj] = var

        check_button.pack()

    # button 1 - ready (calling create_selected_models())
    btn_submit_data_init = tk.Button(init_tab,
                                     text="Ready",
                                     font=("Arial", 20),
                                     command=create_selected_models)
    btn_submit_data_init.pack(anchor="se")

    # ----------------------------------------------------------------------------

    ### main tab (second)
    # canva instance
    canvas = tk.Canvas(main_tab, width=600, height=600, bg="white", cursor="cross")
    canvas.pack(anchor="nw")

    # reseting coordinates from canvas
    canvas.old_coords = None

    # bind mouse functionality to functions
    canvas.bind('<B1-Motion>', draw_line)
    canvas.bind("<ButtonRelease-1>", reset_cords)

    # making sections on main tab
    background_lines = tk.Canvas(main_tab, width=400, height=760, bg="grey94")
    background_lines.create_line(0, 600, 400, 600, fill="black", width=2)
    background_lines.create_line(0, 300, 400, 300, fill="black", width=2)
    background_lines.create_line(0, 150, 400, 150, fill="black", width=2)
    background_lines.create_line(2, 600, 2, 760, fill="black", width=3)
    background_lines.place(x=600, y=0)

    # ----------------------------------------------------------------------------

    ### buttons functions
    def exit_app():
        """Closing program."""

        window.destroy()

    def add_record():
        """Add data to "new_recods.csv" file from canva."""

        # variable for entry
        entry_text = entry_add_record.get()

        # check entered text
        if not entry_text.isnumeric():
            submited_number_text = "wrong"
        elif 0 <= int(entry_text) <= 9:
            submited_number_text = f"Added: {str(entry_text)}"

            gNewRecord.append(entry_text)
            print(f"New record: {entry_text}")

        else:
            submited_number_text = "wrong"

        # label 1, section "Add record"
        lbl_submited_number = tk.Label(main_tab,
                                       text=submited_number_text,
                                       font=("Arial", 20),
                                       width=10,
                                       height=1)
        lbl_submited_number.place(x=780, y=65)

        # get the canva corners for image processing
        x_ = main_tab.winfo_rootx() + canvas.winfo_x()
        y_ = main_tab.winfo_rooty() + canvas.winfo_y()
        x1_ = x_ + canvas.winfo_width()
        y1_ = y_ + canvas.winfo_height()

        # create an instance from canva widget to image object
        img_ = ImageGrab.grab().crop(
            (x_ + 3, y_ + 3, x1_ - 5, y1_ - 3))  # cut the edges because of non zero border width

        # proccess the data
        img_ = img_.resize((28, 28))  # resize the image to 28x28 size
        img_ = img_.convert('L')  # reformat do grayscale
        img_ = PIL_Ops.invert(img_)
        img_ = np.array(img_)

        img_ = img_.reshape(1, 784)
        img_ = img_ / 255.0

        # csv file format:
        # odd number is a row with data from 1 sample, "," separator
        # even number is feature corelated with above sample
        # 1x784 (28x28 captured from square canva)

        # csv writer with append option
        if submited_number_text != "wrong":
            with open("new_records.csv", "a", newline='') as csv_f:
                writer = csv.writer(csv_f)
                writer.writerow(img_[0])
                writer.writerow(entry_text)

        # csv reader to read number of samples
        with open("new_records.csv", "r") as csv_f_:
            reader = csv.reader(csv_f_)
            row_count = sum(1 for row in reader)
            print(f"Rows in csv = {row_count / 2}")

    def clear_canvas():
        """Clear canva to be blank again."""

        canvas.delete("all")

    def show_statistics():
        """Make new window with statistics for selected, trained model."""

        # gSelectedClassifiers[gChosenCLF[0]] == KNeighborsClassifier()
        # gChosenCLF[0] == KNN

        # make variable for model name
        classifier_name = gChosenCLF[0]

        # new window with statistics
        new_window = tk.Toplevel()
        new_window.title("Classification report for " + classifier_name)
        new_window.geometry("510x430")

        # make a variable with report from chosen model
        clf_report = gReports[classifier_name]

        # outside loop with features and averages metric scores with score methods
        for enum1, (feature, score_method) in enumerate(clf_report.items()):
            if not feature == "accuracy":  # if "feature" is not accuracy score_methods are dictionaries

                # column with features and score methods
                lbl_stat_outer = tk.Label(new_window,
                                          text=str(feature),
                                          font=("Arial", 16, "bold"))
                lbl_stat_outer.grid(row=enum1 + 1, column=0)

                # nested loop with score methods with their values
                for enum2, (score_m_, value) in enumerate(score_method.items()):

                    # cell data
                    stats_in_val_lbl = tk.Label(new_window,
                                                text=str(round(value, 2)),
                                                font=("Arial", 16),
                                                padx=5)
                    stats_in_val_lbl.grid(row=enum1 + 1, column=enum2 + 1)

                    # row with precision, f1-score, recall and support
                    if enum1 == 0:
                        stats_in_lbl = tk.Label(new_window,
                                                text=str(score_m_),
                                                font=("Arial", 16, "bold"),
                                                padx=5)
                        stats_in_lbl.grid(row=0, column=enum2 + 1)

            # special occurence with accuracy which is not dictionary type but float
            if feature == "accuracy":  # if "feature" is accuracy score_methods is the value of accuracy

                # "Accuracy"
                lbl_stat_outer = tk.Label(new_window,
                                          text=str(feature),
                                          font=("Arial", 16, "bold"))
                lbl_stat_outer.grid(row=enum1 + 1, column=0)

                # accuracy score
                stats_acc_lbl = tk.Label(new_window,
                                         text=str(score_method),
                                         font=("Arial", 16),
                                         padx=5)
                stats_acc_lbl.grid(row=enum1 + 1, column=1)

    # run once decorator for single use function
    @run_once
    def set_radio_button_init(radio_button):
        radio_button.select()

    def choose_model():
        """Darken not selected models, return model name and model instance "pointer" with list."""

        # loop through selected models from init page
        for rd_btn, (key, val) in zip(radio_buttons, gSelectedClassifiers.items()):
            # if model was selected make it active
            if val:
                rd_btn.config(state=tk.ACTIVE)

                # call a function to select first active radio button
                run_once(set_radio_button_init(rd_btn))

            if not val:
                rd_btn.config(state=tk.DISABLED)

        # option 1 "KNN"
        if model_ind_radio.get() == 0:
            # check if list is empty, if not, make it empty
            if len(gChosenCLF) == 1:
                gChosenCLF.clear()

            # if nothing was selected previously add selected model to list and get it name
            if len(gChosenCLF) == 0:
                clf_name_var = gNames[model_ind_radio.get()]
                gChosenCLF.append(gNames[model_ind_radio.get()])

        # option 2 "SVC"
        elif model_ind_radio.get() == 1:
            if len(gChosenCLF) == 1:
                gChosenCLF.clear()

            if len(gChosenCLF) == 0:
                clf_name_var = gNames[model_ind_radio.get()]
                gChosenCLF.append(gNames[model_ind_radio.get()])

        # option 3 "QDA"
        elif model_ind_radio.get() == 2:
            if len(gChosenCLF) == 1:
                gChosenCLF.clear()

            if len(gChosenCLF) == 0:
                clf_name_var = gNames[model_ind_radio.get()]
                gChosenCLF.append(gNames[model_ind_radio.get()])

        # option 4 "Decision_Tree"
        elif model_ind_radio.get() == 3:
            if len(gChosenCLF) == 1:
                gChosenCLF.clear()

            if len(gChosenCLF) == 0:
                clf_name_var = gNames[model_ind_radio.get()]
                gChosenCLF.append(gNames[model_ind_radio.get()])

        # option 5 "Random_Forest"
        elif model_ind_radio.get() == 4:
            if len(gChosenCLF) == 1:
                gChosenCLF.clear()


            if len(gChosenCLF) == 0:
                clf_name_var = gNames[model_ind_radio.get()]
                gChosenCLF.append(gNames[model_ind_radio.get()])

        # option 6 "Neural_Net"
        elif model_ind_radio.get() == 5:
            if len(gChosenCLF) == 1:
                gChosenCLF.clear()

            if len(gChosenCLF) == 0:
                clf_name_var = gNames[model_ind_radio.get()]
                gChosenCLF.append(gNames[model_ind_radio.get()])

        # option 7 "AdaBoost"
        elif model_ind_radio.get() == 6:
            if len(gChosenCLF) == 1:
                gChosenCLF.clear()

            if len(gChosenCLF) == 0:
                clf_name_var = gNames[model_ind_radio.get()]
                gChosenCLF.append(gNames[model_ind_radio.get()])

        # option 8 "Naive_Bayes"
        elif model_ind_radio.get() == 7:
            if len(gChosenCLF) == 1:
                gChosenCLF.clear()

            if len(gChosenCLF) == 0:
                clf_name_var = gNames[model_ind_radio.get()]
                gChosenCLF.append(gNames[model_ind_radio.get()])

    def submit():
        """Grab canva image, call function to predict, show prediction and probability."""
        # get the canva corners for Image preprocessing
        x = main_tab.winfo_rootx() + canvas.winfo_x()
        y = main_tab.winfo_rooty() + canvas.winfo_y()
        x1 = x + canvas.winfo_width()
        y1 = y + canvas.winfo_height()

        # create an instance from canva widget to image object
        img = ImageGrab.grab().crop((x + 3, y + 3, x1 - 5, y1 - 3))  # cut the edges because of non zero border width

        # show an image with Matplotlib
        # img.show()

        # pass above instance to function for further processing
        prediction, predict_prob = predict_digit(img)

        # label with predicted number
        lbl_result = tk.Label(main_tab,
                              text=str(prediction[0]),
                              font=("Arial", 25),
                              width=2,
                              height=1,
                              bg="white")
        lbl_result.place(x=830, y=320)

        # add all non zero predictions to a list
        max_prob_list = []

        print(f"Probability of each number is:")

        # show non zero predictions
        for ite, prob in enumerate(predict_prob[0]):
            if prob > 0:
                max_prob_list.append(prob)
                print(f"{ite}: {int(prob * 100)}%")

        print()

        # get max propability prediction
        max_prob = max(max_prob_list)

        # show maximum probability prediction
        lbl_result = tk.Label(main_tab,
                              text=str(int(max_prob * 100)) + "%",
                              font=("Arial", 25),
                              width=5,
                              height=1,
                              bg="white")
        lbl_result.place(x=900, y=320)

    def predict_digit(image):
        """Procces the image, return prediction and probability."""

        image = image.resize((28, 28))  # resize the image to dataset size
        image = image.convert('L')  # reformat do grayscale
        image = PIL_Ops.invert(image)
        # image.show()  # show the processed image in new window"""
        image = np.array(image)

        image = image.reshape(1, 784)
        image = image / 255.0

        # make a prediction with model instance
        prediction = gSelectedClassifiers[gChosenCLF[0]].predict(image)

        # get propability with a model instance
        prediction_prob = gSelectedClassifiers[gChosenCLF[0]].predict_proba(image)

        return [prediction, prediction_prob]

    # ----------------------------------------------------------------------------

    # button 1, section "Submit"
    btn_submit = tk.Button(main_tab,
                           text="submit result",
                           font=("Arial", 25),
                           width=10,
                           height=1,
                           command=submit)
    btn_submit.place(x=605, y=310)

    # button 2, section "Show statistics"
    btn_show_data = tk.Button(main_tab,
                              text="show",
                              font=("Arial", 25),
                              width=5,
                              height=1,
                              command=show_statistics)
    btn_show_data.place(x=750, y=220)

    # button 3, section "Add record"
    btn_add = tk.Button(main_tab,
                        text="add",
                        font=("Arial", 25),
                        width=5,
                        height=1,
                        command=add_record)
    btn_add.place(x=615, y=50)

    # button 4, section "Clear and close" - close
    btn_exit = tk.Button(main_tab,
                         text="exit",
                         font=("Arial", 25),
                         width=6,
                         height=1,
                         command=exit_app)
    btn_exit.place(x=860, y=660)

    # button 5, section "Clear and close" - clear
    btn_clear = tk.Button(main_tab,
                          text="clear",
                          font=("Arial", 25),
                          width=6,
                          height=1,
                          command=clear_canvas)
    btn_clear.place(x=720, y=660)

    # ----------------------------------------------------------------------------

    # label 1, section "Show statistics"
    lbl_stats = tk.Label(main_tab,
                         text="Stats",
                         font=("Arial", 25),
                         width=5,
                         height=1)
    lbl_stats.place(x=750, y=160)

    # Label 2, section "Choose model"
    lbl_choose_algoritm = tk.Label(main_tab,
                                   text="Choose model",
                                   font=("Arial", 25),
                                   width=14,
                                   height=1)
    lbl_choose_algoritm.place(x=150, y=610)

    # Label 3, section "Add recod"
    lbl_add_record = tk.Label(main_tab,
                              text="Add record",
                              font=("Arial", 23))
    lbl_add_record.place(x=720, y=10)

    # ----------------------------------------------------------------------------

    # entry 1, section "Add record"
    entry_add_record = tk.Entry(main_tab,
                                font=("Arial", 25),
                                width=2)
    entry_add_record.place(x=730, y=65)

    # ----------------------------------------------------------------------------

    # Tkinter variable
    model_ind_radio = tk.IntVar()

    radio_buttons = []

    # x position iterator
    pos_x = 10

    # variable storing classifier name temporarily
    gClfName = ""

    # list storing classifier instance temporarily
    gChosenCLF = []

    # radiobuttons with selected models names
    for index in range(len(gNames)):
        rbtn_algorithm = tk.Radiobutton(main_tab,
                                        text=gNames[index],
                                        variable=model_ind_radio,
                                        value=index,
                                        padx=1,
                                        pady=2,
                                        font=("Arial", 12),
                                        compound="left",
                                        relief=tk.RAISED,
                                        activebackground="gray100",
                                        command=choose_model)

        radio_buttons.append(rbtn_algorithm)

        # place buttons in 2 row:
        if index < len(gClassifiers) / 2:  # first row placement
            rbtn_algorithm.place(x=pos_x, y=650)

        elif index == len(gClassifiers) / 2:  # second row, first element (reseting x position iterator)
            pos_x = 10
            rbtn_algorithm.place(x=pos_x, y=700)

        elif index >= len(gClassifiers) / 2 + 1:  # second row, other elements
            rbtn_algorithm.place(x=pos_x, y=700)

        pos_x += 145

    # ----------------------------------------------------------------------------

    # main loop with project window
    window.mainloop()


if __name__ == "__main__":
    main()
