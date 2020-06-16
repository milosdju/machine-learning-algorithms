from tkinter import *
import numpy as np

color = None
color_ids = dict()
data = np.empty((0,3), int)

def get_color_id(color):
    if not color_ids:
        color_ids[color] = 1
    elif color not in list(color_ids.keys()):
        color_ids[color] = max(list(color_ids.values())) + 1
    
    return color_ids[color]

def draw_dot(event):
    global data
    x1, y1 = (event.x - 2), (event.y - 2)
    x2, y2 = (event.x + 2), (event.y + 2)
    if color is None: return
    canvas.create_oval(x1, y1, x2, y2, outline = color, fill=color)
    color_id = get_color_id(color)
    row = np.array([event.x, event.y, color_id])
    data = np.vstack([data, row])
    print(data)

def on_box_click(event):
    global color
    rect_id = event.widget.find_closest(event.x, event.y)[0]
    color = canvas.itemcget(rect_id, 'fill')

def save_data(event):
    global data
    if filename_entry.get() is None: return
    np.savetxt(filename_entry.get(), data, delimiter = " ")
    print("Data successfully saved to file %s" % filename_entry.get())

root = Tk()
root.title("Data drawer")

### Filename Entry ###
filename_entry = Entry(root, width = 30, textvariable = "something")
filename_entry.grid(row = 0, column = 0, columnspan = 2, sticky = W, padx = 20, pady = 10)

### Save Button ###
save_button = Button(root, text = "Save")
save_button.grid(row = 0, column = 2, sticky = W)
save_button.bind('<Button-1>', save_data)

canvas = Canvas(root, width = 600, height = 400)
canvas.grid(row = 1, column = 0, columnspan = 5)

### Drawing polygon ###
drawing_polygon = canvas.create_rectangle(20, 20, 500, 380, fill='light grey')

### Color chooser ###
# (x1,y1), (x3,y3)
blue_box = canvas.create_rectangle(520, 20, 540, 40, fill='blue')
red_box = canvas.create_rectangle(550, 20, 570, 40, fill='red')
green_box = canvas.create_rectangle(520, 50, 540, 70, fill='green')
gray_box = canvas.create_rectangle(550, 50, 570, 70, fill='gray')

canvas.tag_bind(drawing_polygon, '<Button-1>', draw_dot)
canvas.tag_bind(blue_box, '<Button-1>', on_box_click)
canvas.tag_bind(red_box, '<Button-1>', on_box_click)
canvas.tag_bind(green_box, '<Button-1>', on_box_click)
canvas.tag_bind(gray_box, '<Button-1>', on_box_click)

root.mainloop()