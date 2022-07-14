from tkinter import *
from videoPredict import start_video
import numpy as np
import json
import matplotlib.pyplot as plt

# create a tkinter window
window = Tk()

# Open window having dimension 100x100
window.geometry('480x480')


def testFunction():
    print("test")


def exit():
    print(window.destroy)


def process_emotion_data(data):
    keys = list(data['emotions'].keys())
    values = list(data['emotions'].values())
    genderKeys = list(data['gender'].keys())
    gender = ''
    if data['gender'][genderKeys[0]] > data['gender'][genderKeys[1]]:
        gender = genderKeys[0]
    else:
        gender = genderKeys[1]

    total = 0
    for v in values:
        total += v
    results = {}
    for index, key in enumerate(keys):
        results[key] = values[index]/total * 100
    return {"emotions": results, "gender": gender}


def read_json():
    f = open('emotion.json')
    data = json.load(f)
    data = process_emotion_data(data)
    return data


def show_graph():
    # creating the dataset
    data = read_json()
    gender = data['gender']
    keys = list(data['emotions'].keys())
    values = list(data['emotions'].values())

    # creating the bar plot
    plt.bar(keys, values, color='maroon',
            width=0.4)

    plt.xlabel("Emotions")
    plt.ylabel("Frequency (%)")
    plt.title("Emotions Capture of " + "(" + gender + ")")
    plt.show()


menu = Menu(window)
new_item = Menu(menu, tearoff=0)
new_item.add_command(label='Open recent', command=show_graph)
menu.add_cascade(label='File', menu=new_item)
menu.add_cascade(label='Exit', command=exit)
window.config(menu=menu)

# Create a Button
btn = Button(window, text='Start', bd='5',
             command=start_video)

# Set the position of button on the top of window.
btn.pack(side="top")

window.mainloop()
