function [testData, testLabels] = preprocessTestData(testBatchFileName, imageSize)
    % Load the test batch using loadCIFAR10Batch function
    [testImages, testLabels] = loadCIFAR10Batch(testBatchFileName);
    
    % Initialize the processed test data array
    numTestImages = 64;
    testData = zeros([imageSize, numTestImages], 'uint8');

    % Resize and normalize the test images
    for idx = 1:numTestImages
        % Resize image
        resizedImage = imresize(testImages(:, :, :, idx), imageSize(1:2));
        
        % Convert uint8 to double for normalization
        resizedImage = double(resizedImage);
        
        % Normalize image
        meanRGB = [123.68, 116.779, 103.939];
        testData(:, :, :, idx) = resizedImage - reshape(meanRGB, [1 1 3]);
    end
    
    % Convert labels to categorical for the classification task
    testLabels = categorical(testLabels);
end
