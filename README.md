MATLAB MNIST Digit Recognizer (Custom IDX Reader + CNN)

This project implements a complete handwritten digit recognition system in MATLAB using a custom IDX parser and a Convolutional Neural Network (CNN).
You manually load the MNIST dataset from raw .idx files and train a deep learning model without relying on built-in MNIST helper functions.

The workflow includes:

Reading MNIST image and label files in IDX binary format

Validating headers and converting data into MATLAB arrays

Preparing training data (XTrain, YTrain)

Creating a CNN architecture

Training the network using trainNetwork

📌 Features

Fully manual MNIST .idx file reader (no external scripts required)

Error-checked image and label parsing

Automatic conversion into 4-D CNN-ready format

Clean and simple CNN architecture

Uses Deep Learning Toolbox (trainNetwork, imageInputLayer, etc.)

Training progress visualization

📂 Project Structure
|-- your_script.m          % Final code provided in this repository
|-- t10k-images.idx3-ubyte % MNIST test images
|-- t10k-labels.idx1-ubyte % MNIST test labels
|-- README.md
|-- LICENSE
🧠 How It Works
1. Read MNIST images

Opens file in big-endian mode

Validates magic number (2051)

Reads counts, dimensions, and individual pixel data

Converts each image into a 28×28 matrix

Stores images in imageCellArray

2. Read MNIST labels

Validates magic number (2049)

Reads labels one by one

Stores in labelCellArray

3. Convert to 4-D CNN format
XTrain(:,:,1,i) = imageCellArray{i};
YTrain = categorical(labels);
4. CNN Architecture

A simple but effective model:

Input layer (28×28×1)

Two convolution + batch normalization + ReLU blocks

Max pooling

Fully connected layer (10 classes)

Softmax + classification layer

5. Training

Configured with:

Stochastic Gradient Descent with Momentum (SGDM)

4 epochs

Learning rate: 0.01

Training progress appears automatically.

▶️ Running the Project

Place MNIST files (t10k-images.idx3-ubyte and t10k-labels.idx1-ubyte) in the project folder.

Then run the script:

run your_script.m

After training, the network is stored in:

net

You can classify any processed image using:

label = classify(net, XTrain(:,:,:,i));
🏁 Result

This setup typically achieves high accuracy on MNIST using only a small CNN and minimal training time.

📄 License

This project is under the MIT License (see LICENSE file).

🎯 Future Enhancements

Add real-time digit prediction from webcam

Add GUI for drawing digits

Test/train split using the full MNIST dataset

Replace CNN with deeper architecture (LeNet-5 / VGG-like)
