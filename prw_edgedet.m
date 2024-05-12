function prw_edgedet()
    % Prewitt operator kernels
    prw_x = [-1, 0, 1; -1, 0, 1; -1, 0, 1]; % Prewitt Gx
    prw_y = [-1, -1, -1; 0, 0, 0; 1, 1, 1]; % Prewitt Gy
    
    % Load image
    [filename, pathname] = uigetfile({'*.png'}, 'Select an image');
    imagename = fullfile(pathname, filename);
    im = imread(imagename);
    
    % Convert to grayscale if necessary
    if size(im, 3) == 3
        gray_im = rgb2gray(im); % Convert RGB image to grayscale
    else
        gray_im = im; % Image is already grayscale
    end
    
    % Convolve image with Prewitt kernels to get IX and IY arrays
    edge_x = conv2(double(gray_im), prw_x, 'same'); % Convolution with Prewitt Gx
    edge_y = conv2(double(gray_im), prw_y, 'same'); % Convolution with Prewitt Gy
    
    % Calculate gradient magnitude array from IX and IY arrays
    gradient_mag = sqrt(edge_x .^ 2 + edge_y .^ 2);
    
    % Calculate gradient direction
    gradient_dir = atan2d(edge_y, edge_x);
    gradient_dir(gradient_dir < 0) = gradient_dir(gradient_dir < 0) + 180; % Ensure gradient direction is within [0, 180] degrees
    
    % Non-maximum suppression to thin the edges
    nms = non_maxima_suppress_custom(gradient_mag, gradient_dir);
    
    % Display final result after NMS
    figure, imshow(uint8(nms)), title('Final result after NMS');
    
    % Save the output image
    [~, name, ~] = fileparts(filename);
    output_filename = fullfile(pathname, [name '_output.png']);
    imwrite(uint8(nms), output_filename);
end

function nms = non_maxima_suppress_custom(im, angle)
    nms = zeros(size(im));
     % iterates through each pixel of the input image except for the border pixels
    for y = 2:size(im, 1) - 1
        for x = 2:size(im, 2) - 1
           %  For each pixel, if its gradient magnitude is not the maximum among its neighbors in the edge direction, its value is set to zero in the output image
           %  Otherwise, it retains its original magnitude.
            if (angle(y, x) >= 0 && angle(y, x) < (pi/8))
                neighbour_value = max(im(y, x-1), im(y, x+1));
            elseif (angle(y, x) >= (pi/8) && angle(y, x) < 67.5)
                neighbour_value = max(im(y-1, x-1), im(y+1, x+1));
            elseif (angle(y, x) >= 67.5 && angle(y, x) < 112.5)
                neighbour_value = max(im(y-1, x), im(y+1, x));
            elseif (angle(y, x) >= 112.5 && angle(y, x) < pi/1.142)
                neighbour_value = max(im(y-1, x+1), im(y+1, x-1));
            else
                neighbour_value = max(im(y, x-1), im(y, x+1));
            end
            if im(y, x) < neighbour_value
                nms(y, x) = 0; % if M(A) > M(C) or M(B) > M(C) then set M(x, y) = 0
            else
                nms(y, x) = im(y, x);
            end
        end
    end
end
