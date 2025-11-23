

clearvars; close all; clc

%% FOR READING IMAGES

filename_image = 't10k-images.idx3-ubyte';

fid1 = fopen(filename_image, 'r', 'ieee-be');
if fid1 == -1
    error('Could not open file.');
end

magicNumber_image = fread(fid1, 1, 'uint32');

if magicNumber_image ~= 2051
    error('Unexpected magic number: %d (expected 2051).', magicNumber_image);
end

totalImages = fread(fid1, 1, 'uint32');
numRows     = fread(fid1, 1, 'uint32');
numCols     = fread(fid1, 1, 'uint32');

% Validate header numbers
if totalImages <= 0 || numRows <= 0 || numCols <= 0
    error('Invalid header values read from file.');
end

% Preallocate cell array for images
imageCellArray = cell(totalImages, 1);

% Read each image
pixelsPerImage = double(numRows) * double(numCols);
for k = 1:totalImages
    % Read pixels as uint8. Use 'uint8=>uint8' to get uint8 output directly.
    A = fread(fid1, pixelsPerImage, 'uint8=>uint8');
    if numel(A) ~= pixelsPerImage
        error('Unexpected end of file while reading image %d/%d.', k, totalImages);
    end
    % Reshape into matrix and transpose to get the correct orientation
    img = reshape(A, numCols, numRows)';   % results in numRows x numCols
    imageCellArray{k} = img;
end


%% FOR READING LABLES

filename_label = 't10k-labels.idx1-ubyte';

fid2 = fopen(filename_label, 'r', 'ieee-be');
if fid2 == -1
    error('Could not open file.');
end

magicNumber_label = fread(fid2, 1, 'uint32');

if magicNumber_label ~= 2049
    error('Unexpected magic number: %d (expected 2049).', magicNumber_label);
end

totalLabels = fread(fid2, 1, 'uint32');

% Validates if Total Images is Equal to Total Lables
if totalLabels ~= totalImages
    error('Total Lable does not Match Total Images ');
end

% Preallocate cell array for label
labelCellArray = cell(totalLabels, 1);

for k = 1:totalLabels
    % Read label as uint8. Use 'uint8=>uint8' to get uint8 output directly.
    B = fread(fid2, 1, 'uint8=>uint8');
    labelCellArray{k}= B;
end

%% XTrain and YTrain
numImages = numel(imageCellArray);
imgSize = size(imageCellArray{1});   % should be 28x28

XTrain = zeros(imgSize(1), imgSize(2), 1, numImages, 'uint8');

for i = 1:numImages
    XTrain(:,:,1,i) = imageCellArray{i};
end

labels = cell2mat(labelCellArray);  % converts cell → numeric vector
YTrain = categorical(labels);

%% Layers

layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];

%% Options

options = trainingOptions('sgdm', ...
    'MaxEpochs',4, ...
    'InitialLearnRate',0.01, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Train

net = trainNetwork(XTrain, YTrain, layers, options);

