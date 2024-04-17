function edgeFeatures = cannyEdgeFeatures(img)

    grayImg = rgb2gray(img);
    edges = edge(grayImg, 'Canny');
    % Flatten the edge image into a feature vector
    edgeFeatures = edges(:)';
end