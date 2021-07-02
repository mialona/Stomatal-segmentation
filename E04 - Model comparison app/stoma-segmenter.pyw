from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk
from PIL import ImageTk, Image
from fastai.basics import *
import torchvision.transforms as transforms

# Lists with the names of the models and their directories.
model_names = []
model_dirs = []

# Reading the file and loading the lists of model names and their directories.
text_file = open("models.txt", "r").read()
for pair in text_file.split("\n"):
    aux = pair.split(",")
    model_names.append(aux[0])
    model_dirs.append(aux[1])

# Interface configuration.
root = Tk()
root.title("Stoma segmenter")
root.iconphoto(False, PhotoImage(file='icon.png'))
root.geometry("825x300+300+150")
root.resizable(width=False, height=False)
root.configure(background='#303030')

# Initiation of global variables.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img = None
img_h = None
img_w = None
mask = None
panel1 = None
panel2 = None

# Displays a dialog box that allows the user to choose the image to open and
# returns the file path.
def open_file_name():
    filename = filedialog.askopenfilename()
    return filename

# Displays a dialog box that allows the user to choose where to save the
# image (PNG) and returns the file path.
def save_file_name():
    filename = filedialog.asksaveasfilename(filetypes=(
                    ("Portable Network Graphics", "*.png"),
                ),defaultextension = ".png")
    return filename

# Loading the model of the given path.
def load_model(modelName):
    return torch.jit.load(modelName)

# Normalizes the given image and converts it to a tensor.
def transform_image(image):
    global device

    my_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image_aux = image
    return my_transforms(image_aux).unsqueeze(0).to(device)

# Load an image in the global variable 'img' with dimensions 50x50 px and
# display it in the first panel. Displays an error message in case of failure.
def open_img():
    global img
    global img_h
    global img_w
    global panel1
    global panel2

    try:
        x = open_file_name()

        if (x != ""):
            img = Image.open(x)
            img_h = img.height
            img_w = img.width
            img = img.resize((50, 50), Image.ANTIALIAS)
            imgTk = ImageTk.PhotoImage(img.resize((250, 250), Image.ANTIALIAS))

            if(panel1 is not None):
                panel1.destroy()
            if(panel2 is not None):
                panel2.destroy()

            panel1 = Label(root, image=imgTk, bg="black")
            panel1.image = imgTk
            panel1.place(x=250, y=25)
    except:
        messagebox.showinfo(message="Error loading image.", title="Error")

# Saves the generated mask in a file (PNG). Displays an error message in case
# of failure.
def save_img():
    global mask
    global img_h
    global img_w

    try:
        x = save_file_name()

        if (x != ""):
            mask.resize((img_w, img_h), Image.ANTIALIAS).save(x)
    except:
        messagebox.showinfo(message="Error saving mask.", title="Error")

# Given the name of a model, it generates the corresponding mask of the
# current image in array format.
def do_mask(model_name):
    global img
    global device

    tensor = transform_image(image=img)

    model = load_model("models/"+model_name)
    model = model.cpu()
    model.to(device)
    with torch.no_grad():
        outputs = model(tensor)

    outputs = torch.argmax(outputs,1)

    return np.array(outputs.cpu())

# From the models chosen in the comboboxes and the selected operation, it
# generates the masks, transforms them to an image and places the latter in
# the second frame. Displays an error message in case of failure.
def annotation(option,CB_first,CB_second):
    global mask
    global panel2

    try:
        if(CB_first.current() == -1):
            messagebox.showinfo(message="Model not selected.", title="Error")
            return
        mask1 = do_mask(model_dirs[CB_first.current()])

        # Flow according to the selected operation.
        if (option.get() == 0):
            mask1[mask1==1]=255
            mask1 = np.reshape(mask1,(50,50))
            mask = Image.fromarray(mask1.astype('uint8'))
        else:
            if(CB_second.current() == -1):
                messagebox.showinfo(message="Model not selected.", title="Error")
                return
            mask2 = do_mask(model_dirs[CB_second.current()])

            if (option.get() == 1):
                mask_aux = np.logical_and(mask1,mask2)
            else:
                mask_aux = np.logical_or(mask1,mask2)

            mask_aux = mask_aux.astype(int)
            mask_aux[mask_aux==True]=255
            mask_aux = np.reshape(mask_aux,(50,50))
            mask = Image.fromarray(mask_aux.astype('uint8'))

        # Mask transformation to RGB.
        rgbimg = Image.new("RGB", mask.size)
        rgbimg.paste(mask)
        rgbimg = np.array(rgbimg)

        # Mask coloring.
        rgbimg[np.where((rgbimg==[255,255,255]).all(axis=2))] = [176,28,20]
        rgbimg[np.where((rgbimg==[0,0,0]).all(axis=2))] = [255,255,255]

        # Mask overlay on image.
        aux = np.array(img)
        out_img = np.zeros(aux.shape, dtype=aux.dtype)
        out_img[:,:,:] = (0.5 * aux[:,:,:]) + (0.5 * rgbimg[:,:,:])
        out_img = Image.fromarray(out_img)

        imgTk = ImageTk.PhotoImage(out_img.resize((250, 250), Image.ANTIALIAS))

        if(panel2 is not None):
            panel2.destroy()

        panel2 = Label(root, image=imgTk, bg="black")
        panel2.image = imgTk
        panel2.place(x=550, y=25)
    except:
        messagebox.showinfo(message="Error creating mask.", title="Error")


# Creation and placement of interface controls.
# Buttons.
B_open = Button(root, text='Open image', command=open_img).place(x=25, y=40)
B_save = Button(root, text='Save mask', command=save_img).place(x=135, y=40)

B_annotate = Button(root, text='Annotate', command=lambda:annotation(option,CB_first,CB_second)).place(x=140, y=225)

# Radiobuttons.
option = IntVar()
RB_none = Radiobutton(root, text="None",command=lambda:CB_second.config(state="disable"),
                      variable=option, value=0,
                      background='#303030', fg="white",
                      activebackground='#303030', activeforeground="white",
                      selectcolor="gray").place(x=25, y=100)
RB_and = Radiobutton(root, text="And",command=lambda:CB_second.config(state="readonly"),
                      variable=option, value=1,
                      background='#303030', fg="white",
                      activebackground='#303030', activeforeground="white",
                      selectcolor="gray").place(x=85, y=100)
RB_or = Radiobutton(root, text="Or",command=lambda:CB_second.config(state="readonly"),
                      variable=option, value=2,
                      background='#303030', fg="white",
                      activebackground='#303030', activeforeground="white",
                      selectcolor="gray").place(x=145, y=100)

# Comboboxes.
CB_first = ttk.Combobox(root, state="readonly", width=25)
CB_first["values"] = model_names
CB_first.place(x=25, y=145)

CB_second = ttk.Combobox(root, state="disabled", width=25)
CB_second["values"] = model_names
CB_second.place(x=25, y=175)

# Frames.
Frame(root, width=250, height=250, bg="black").place(x=250, y=25)
Frame(root, width=250, height=250, bg="black").place(x=550, y=25)

# Main loop of the application.
root.mainloop()