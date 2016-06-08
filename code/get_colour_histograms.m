function [ colour_histograms ] = get_colour_histograms( images_paths,quantisationLevel,colourSpace, HIST_NORMALISE )
%GET_COLOUR_HISTOGRAMS Returns quantised colour histograms in a vectorised form
%   output:
%       colour_histograms =  an N x d matrix of colour histograms
%                            images. E.g. if the quantisationLevel is 16, d would equal 16*16*16.
%   parameters:
%       images_path =        an N x 1 cell array of strings where each string is an
%                            image path on the file system.
%       quantisationLevel =  level of quantisation applied to each image
%                            Directly corresponds to featureSize which is
%                            quantisationLevel cubed.
%       colourSpace =        image is converted to chosen colour space before
%                            the hitogram is obtained. Options:
%                            'ycbcr', 'rgb', 'lab', 'opponent'*, 'hsv',
%                            'hsv-hue'
%
% *defined in Ballard, D.H., and Brown, C.M. 1982. Computer Vision. Prentice
%  Hall: New York.


noImages = length(images_paths);
noBins = quantisationLevel;
featureSize = noBins*noBins*noBins;
hsvHue = false;

if strcmp(colourSpace,'hsv-hue')
    colour_histograms = zeros(noImages,noBins);
else
    colour_histograms = zeros(noImages,featureSize);
end


for i=1:noImages
    img = imread(images_paths{i});
    
    switch colourSpace
        case 'rgb'
            % do nothing, images are already in rgb
        case 'ycbcr'
            img = rgb2ycbcr(img);
        case 'lab'
            colorTransform = makecform('srgb2lab');
            img = applycform(img,colorTransform);
        case 'opponent'
            r = img(:,:,1);
            g = img(:,:,2);
            b = img(:,:,3);
            
            rg = r - g;
            by = 2* b - r - g;
            wb = r + g + b;
            
            img(:,:,1) = rg;
            img(:,:,2) = by;
            img(:,:,3) = wb;
        case 'hsv-full'
            img = rgb2hsv(img);
        case 'hsv-hue'
            img = rgb2hsv(img);
            hsvHue = true;
            
    end
    
    img = double(img);
    
    if strcmp(colourSpace,'hsv-hue') || strcmp(colourSpace,'hsv-full')
        imquant = img; %result is already in 0-1 
    else
        imquant = img/255;
    end
        
    
    imquant = round(imquant*(noBins-1)) + 1;
    
    %%imshow((imquant-1)/(noBins-1));
    if(hsvHue)
        hh=zeros(noBins,1);
    else
        hh = zeros(noBins,noBins,noBins);
    end
    
    
    if(hsvHue)
        imquant = imquant(:,:,1);
        imquantFlat = imquant(:);
        for j=1:size(imquantFlat,1)
            value = imquantFlat(j);
            hh(value) = hh(value) + 1;
        end
        
        histogram = hh;
    else
        
        % flatten the matrix (from A x B x C to AB x C)
        imquantFlat = reshape(imquant,size(imquant,1)*size(imquant,2),size(imquant,3));
        
        for j=1:size(imquantFlat,1)
            colourPerPixel = imquantFlat(j,:);
            hh(colourPerPixel(1),colourPerPixel(2),colourPerPixel(3)) = hh(colourPerPixel(1),colourPerPixel(2),colourPerPixel(3)) + 1;
        end
        
        
        % flatten histogram
        histogram = reshape(hh,size(hh,1)*size(hh,2)*size(hh,3),1);
        
    end
    
    % normalise
    histogram = histogram / sum(histogram(:));
    
    
    colour_histograms(i,:) = histogram';
    
    
end


end


