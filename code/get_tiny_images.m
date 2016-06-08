function [ tiny_images ] = get_tiny_images( images_paths,dimensionSize, METHOD, NORMALISE, COLOUR)

%GET_TINY_IMAGES Returns resized square images in a vectorised form
%   output:
%       tiny_images =   an N x d matrix of resized and then vectorized tiny
%                       images. The d is based on dimensionSize and COLOUR.
%                       E.g when dimensionSize = 16, COLOUR = 'rgb'
%                       d = 16 x 16 x 3
%                       For COLOUR ='greyscale'
%                       d = 16 x 16 x 1
%               
%   parameters:
%       images_path =   an N x 1 cell array of strings where each string is an
%                       image path on the file system.
%       dimensionSize = dimension size of a tiny image after resizing. E.g.
%                       in case of a 16x16 dimensionSize is 16. Feature
%                       size is dimensionSize * dimensionSize
%       method =        method to use when resizing. Can be either
%                       'center-crop' or 'fit'. 'center-crop' will first
%                       crop the original picture to a square whose center
%                       aligns with the picture's center
%                       'fit' will resize the original picture ignoring
%                       original aspect ratio
%       normalise =     'unit variance' - all images normalised to zero mean
%                       unit variance. 
%                       'unit_length' - all images normalised to zero mean
%                       and unit length
%                       'none' - normalisation is performed
%       colour =        'greyscale', 'rgb'

display('Getting tiny images');
noImages = length(images_paths);


switch(COLOUR)
    case 'greyscale'
        planes = 1;
    case 'rgb'
        planes = 3;
end
featureSize = dimensionSize * dimensionSize * planes;
dimensions = [dimensionSize,dimensionSize];

tiny_images = zeros(noImages,featureSize);

for i=1:noImages
    img = imread(images_paths{i});
    
    %   GREYSCALE OR RGB
    switch COLOUR
        case 'greyscale'
            img = rgb2gray(img);
        case 'rgb'
            % do nothing, the image is already colour
    end
    
    %   choose method of resizing
    switch METHOD
        case 'center-crop'
            % create square that is centered and fills
            % the most of the original image
            
            [y,x,planes] = size(img);
            
            % not a square - crop
            if x~=y
                if x<y
                    squareSize = x;
                    bigger = y;
                else
                    squareSize = y;
                    bigger = x;
                end
            
            
            center = [x/2,y/2];
            
            if bigger == x
            topLeftCorner = [center(1)-squareSize/2,1];
            else
            topLeftCorner = [1,center(2)+squareSize/2];
            end
            
            cropArea = [topLeftCorner(1),topLeftCorner(2),squareSize,squareSize];
            
            imgMod = imcrop(img,cropArea);
            imgMod = imresize(imgMod,dimensions);
            else
                %already a square - just resize
                imgMod = imresize(img,dimensions);
            end
            
            
           
                        
        case 'fit'
            imgMod = imresize(img,dimensions);
                        
    end
    
    %   vectorise
    imgMod = imgMod(:);
    
    %   cast to double
    imgMod = double(imgMod);
    
    %   normalising
    switch NORMALISE
        case 'unit-variance'
            % zero mean
            imgMod = imgMod - mean2(imgMod);
            % unit variance
            imgMod = imgMod/std2(imgMod);
            
        case 'unit-length'
            % zero mean
            imgMod = imgMod - mean2(imgMod);
            % unit length
            imgMod = imgMod/norm(imgMod);
        case 'none'
            % no normalisation
    end
    
    tiny_images(i,:) = imgMod';
    
    
end

