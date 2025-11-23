# MATLAB MNIST Digit Recognizer (Custom IDX Reader + CNN)

This project implements a complete handwritten digit recognition system in MATLAB using a **custom IDX parser** and a **Convolutional Neural Network (CNN)**.
The MNIST dataset is loaded manually from raw `.idx` files, validated, converted into arrays, and trained using MATLAB’s Deep Learning Toolbox.

---

## 📌 Features

* Manual MNIST `.idx` file reader (no helper scripts required)
* Error-checked parsing for images and labels
* Converts data into 4-D CNN-ready format (`28×28×1×N`)
* Lightweight and clean CNN architecture
* Uses Deep Learning Toolbox components (`trainNetwork`, `imageInputLayer`, etc.)
* Includes training progress visualization

---

## 📂 Project Structure

```
|-- your_script.m          % Final MATLAB code
|-- t10k-images.idx3-ubyte % MNIST image file
|-- t10k-labels.idx1-ubyte % MNIST label file
|-- README.md
|-- LICENSE
```

---

## 🧠 How It Works

### **1. Read MNIST Images**

* Opens the file in big-endian mode
* Confirms the image magic number (2051)
* Reads total images, rows, and columns
* Loads each image as a `28×28` matrix
* Stores all images in `imageCellArray`

### **2. Read MNIST Labels**

* Confirms the label magic number (2049)
* Ensures number of labels = number of images
* Stores each label in `labelCellArray`

### **3. Convert to CNN Format**

```matlab
XTrain(:,:,1,i) = imageCellArray{i};
YTrain = categorical(labels);
```

### **4. CNN Architecture**

* Input layer (28×28×1)
* Conv → BatchNorm → ReLU
* Max pooling
* Conv → BatchNorm → ReLU
* Fully connected (10 classes)
* Softmax + classification layer

### **5. Training Configuration**

* Optimizer: SGDM
* Epochs: 4
* Learning rate: 0.01
* Training-progress plot enabled

---

## ▶️ Running the Project

1. Place both MNIST files in your project folder:

   * `t10k-images.idx3-ubyte`
   * `t10k-labels.idx1-ubyte`

2. Run the main script:

```matlab
run your_script.m
```

3. The trained network will be stored in:

```matlab
net
```

4. Classify any image like this:

```matlab
label = classify(net, XTrain(:,:,:,i));
```

---

## 🏁 Result

This minimal CNN achieves strong accuracy on MNIST despite its compact size and short training time.

---

## 📄 License

This repository is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🎯 Future Enhancements

* Add webcam-based real-time digit prediction
* Add a GUI to draw digits and classify in real-time
* Add full training/testing split using all MNIST files
* Upgrade to deeper architectures (LeNet-5, VGG-like, etc.)

---
