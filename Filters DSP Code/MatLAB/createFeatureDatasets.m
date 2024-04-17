function [originalDatasetResized, histDataset, cannyDataset, contourDataset] = createFeatureDatasets(batchFileName, imageSize)
    % Load the CIFAR-10 batch using the provided function
    [images, labels] = loadCIFAR10Batch(batchFileName);

    % Initialize the datasets
    numImages = 64;
    originalDatasetResized = zeros([imageSize, numImages], 'uint8');
    histDataset = zeros([imageSize, numImages], 'uint8');
    cannyDataset = zeros([imageSize, numImages], 'uint8');
    contourDataset = zeros([imageSize, numImages], 'uint8');

    % Loop over all images in the batch
    for idx = 1:numImages
        img = images(:,:,:,idx);

        % Resize original image for VGG-16
        originalImg = imresize(img, imageSize(1:2));
        originalDatasetResized(:,:,:,idx) = originalImg;

        % Color Histogram (Equalize each channel)
        histImg = img;
        for channel = 1:3
            histImg(:,:,channel) = histeq(img(:,:,channel));
        end
        histImg = imresize(histImg, imageSize(1:2));
        histDataset(:,:,:,idx) = histImg;

        % Canny Edge Detection
        cannyImg = rgb2gray(img);
        cannyImg = edge(cannyImg, 'Canny');
        cannyImg = imresize(uint8(cannyImg) * 255, imageSize(1:2)); % Rescale to 0-255 and resize
        cannyDataset(:,:,1,idx) = cannyImg; % Copy edge features into all channels
        cannyDataset(:,:,2,idx) = cannyImg;
        cannyDataset(:,:,3,idx) = cannyImg;

        % Contour Detection
        contourImg = rgb2gray(img);
        [~, threshold] = edge(contourImg, 'sobel');
        fudgeFactor = 0.5;
        contourImg = edge(contourImg, 'sobel', threshold * fudgeFactor);
        contourImg = imresize(uint8(contourImg) * 255, imageSize(1:2)); % Rescale to 0-255 and resize
        contourDataset(:,:,1,idx) = contourImg; % Copy contour features into all channels
        contourDataset(:,:,2,idx) = contourImg;
        contourDataset(:,:,3,idx) = contourImg;
    end
end
