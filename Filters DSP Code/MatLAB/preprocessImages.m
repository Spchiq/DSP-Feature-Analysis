function processedImages = preprocessImages(images)
    % Ensure that the images are in double format for processing
    images = double(images);

    % Reshape meanRGB to enable broadcasting
    meanRGB = reshape([123.68, 116.779, 103.939], [1, 1, 3]);

    % Subtract the meanRGB from images
    processedImages = bsxfun(@minus, images, meanRGB);
end
