function [images, labels] = loadCIFAR10Batch(batchFileName)
    % Load the .mat file
    loadedBatch = load(batchFileName);

    % Extract images and labels
    images = loadedBatch.data;
    labels = loadedBatch.labels;

    % Reshape the images if necessary
    % The CIFAR-10 dataset images are 32x32 pixels with 3 color channels
    % Data is stored in row so we reshape it to a 4D array
    images = reshape(images', [32, 32, 3, length(labels)]);
    images = permute(images, [2, 1, 3, 4]); % Transpose the first two dimensions

    % Convert labels to double for consistency
    labels = double(labels);
end
