import torch
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torchvision.transforms as transforms
import torch.nn.functional as F
from tkinter import ttk

model = torch.load('best.pt')
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to predict the image
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    img_t = transform(image)
    img_t = img_t.unsqueeze(0)

    with torch.no_grad():
        output = model(img_t)
        probabilities = F.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        accuracy = probabilities[predicted_class].item() * 100

    return predicted_class, accuracy

# GUI to select image and display results
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).resize((200, 200))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

        predicted_class, accuracy = predict_image(file_path)
        result_label.config(text=f'Predicted Class: {predicted_class}\nAccuracy: {accuracy:.2f}%')

# Set up the GUI
root = tk.Tk()
root.title('Traffic Sign detection')
root.geometry('700x500')
root.configure(bg='#f0f0f0')

style = ttk.Style()
style.configure('TButton', font=('Helvetica', 12), padding=10)
style.configure('TLabel', font=('Helvetica', 12), background='#f0f0f0')

frame = ttk.Frame(root, padding=20)
frame.pack(pady=20)

# Button to select image
button = ttk.Button(frame, text='Select Image', command=open_file)
button.pack(pady=10)

# Label to show selected image
image_label = ttk.Label(frame)
image_label.pack(pady=10)

# Label to show prediction result
result_label = ttk.Label(frame, text='Predicted Class: \nAccuracy: ')
result_label.pack(pady=10)

root.mainloop()
