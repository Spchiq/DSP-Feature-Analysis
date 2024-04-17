%Load CIFAR-10 and create Feature Selection Datasets
[originalResized, histDataset, cannyDataset, contourDataset] = createFeatureDatasets('data_batch_1.mat', [224, 224, 3]);
[data, labels] = loadCIFAR10Batch('data_batch_1.mat');

%Display 16 random images with their labels
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"];
displayImageGrid(data, labels, labelNames, 16);

% Preprocess all datasets
originalDatasetPreprocessed = preprocessImages(originalResized);
histDatasetPreprocessed = preprocessImages(histDataset);
cannyDatasetPreprocessed = preprocessImages(cannyDataset);
contourDatasetPreprocessed = preprocessImages(contourDataset);

%Create a new Model (VGG16)
vgg16Model = vgg16;
layersTransfer = vgg16Model.Layers(1:end-3);

% Define new layers for CIFAR-10
numClasses = numel(unique(labels));
newLayers = [
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];

% Combine the layers
vgg16CIFAR10 = [
    layersTransfer
    newLayers
];

%Learning Settings
opts = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationFrequency', 3, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%Making sure labels are categorical and truncating for batch size
categoricalLabels = categorical(labels);
categoricalLabels = categoricalLabels(1:64);

% Display the number of images and labels to check they match
disp(['Number of images: ', num2str(size(histDatasetPreprocessed, 4))]);
disp(['Number of labels: ', num2str(numel(categoricalLabels))]);

%Train Model
[trainedModel, trainInfo] = trainNetwork(histDatasetPreprocessed, categoricalLabels, vgg16CIFAR10, opts);

%PreProcess TestData (Truncate Testing set for smaller batch size as well)
[testData, testLabels] = preprocessTestData('test_batch.mat', [224, 224, 3]);
testLabels = testLabels(1:64);

%Predict labels with trained Model
predictedLabels = classify(trainedModel, testData);

%Display accuracy from Testing Dataset
accuracy = sum(predictedLabels == testLabels) / numel(testLabels);
disp(['Test accuracy: ', num2str(accuracy)]);
