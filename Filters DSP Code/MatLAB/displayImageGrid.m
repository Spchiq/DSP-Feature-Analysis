function displayImageGrid(images, labels, labelNames, numImages)
    figure;
    numRows = ceil(sqrt(numImages));
    numCols = ceil(numImages / numRows);
    for i = 1:numImages
        subplot(numRows, numCols, i);
        axis off; % Turn off axis
        imshow(images(:,:,:,i));
        title(char(labelNames(labels(i)+1))); % labels are 0-indexed
    end
    sgtitle('CIFAR-10 Dataset');
end

%Run this before to define the labelNames
% labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"];

% displayImageGrid(data, labels, labelNames, 16); 
