
function image_feats = get_hogs(image_paths,vocab_size,colour,smoothing, sigma_smooth,cell_size)


load('vocab.mat')

noImages = length(image_paths);

%% temporary histogram for one image
imgHist = zeros(vocab_size,1);

image_feats = [];


for i=1 : noImages
    img = imread(image_paths{i});
     %% convert to SINGLE
    img = single(img);
    
    if(strcmp(colour, 'greyscale'))
        %% convert to greyscale
        img = rgb2gray(img);
    elseif (strcmp(colour, 'hsv'))
        %% convert to hsv
        img = rgb2hsv(img);
    elseif(strcmp(colour,'w-sift'))
        r = img(:,:,1);
        g = img(:,:,2);
        b = img(:,:,3);
        
        o1 = (r-g)/sqrt(2);
        o2 = (r+g-2*b)/sqrt(6);
        o3 = (r+ g + b)/sqrt(3);
        
        W1 = o1./o3;
        W2 = o2./o3;
        
        img = cat(3,W1,W2);
    end
    
   
    
    %% crop to get constant number of features
    img = cropResize(img,200,200);
    
    
    %% smoothing
    if(smoothing == 1)
        img = vl_imsmooth(double(img), sigma_smooth);
    end
    
    %% extract HOG features
    hog = vl_hog(single(img),cell_size);
    
    hog = hog(:);
    
    %% Local clustering
    D = vl_alldist2(vocab,single(hog));
    
    
    [mindist,indices] = min(D,[],1);
    
    %% build histogram
    for k = 1 : size(mindist,2)
        imgHist(indices(k)) = imgHist(indices(k))+1;
    end
    
    %% normalise histogram
    imgHistNorm = imgHist/sum(imgHist);
    
    imgHistNorm = imgHistNorm';
    %add histogram to feature list
    image_feats = [image_feats;imgHistNorm];
    
end


end

