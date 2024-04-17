function histFeatures = colorHistogramFeatures(img)
    % Create histogram for eeach color then Flatten them into a single feature vector
    redHist = imhist(img(:,:,1), 256)';
    greenHist = imhist(img(:,:,2), 256)';
    blueHist = imhist(img(:,:,3), 256)';
    histFeatures = [redHist, greenHist, blueHist];
end
