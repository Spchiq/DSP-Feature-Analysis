function shape(img, displayFigure)
    % Convert image to grayscale and detect edges
    grayImg = rgb2gray(img);
    edges = edge(grayImg, 'Canny');

    % Find contours using the 'bwboundaries' function
    [B, L] = bwboundaries(edges, 'noholes');

    % Optionally overlay contours on the original image
    if displayFigure
        figure;
        imshow(label2rgb(L, @jet, [.5 .5 .5]));
        hold on;
        for k = 1:length(B)
            boundary = B{k};
            plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2);
        end
        title('Contours');
    end
end
