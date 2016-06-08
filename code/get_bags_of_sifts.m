
function image_feats = get_bags_of_sifts(image_paths,vocab_size,colour,smoothing, sigma_smooth,step, bin_size)


load('vocab.mat')

imgNo = length(image_paths);

%% temporary histogram for one image
imgHist = zeros(vocab_size,1);

image_feats = [];

for i =1 :imgNo
    
    img = imread(image_paths{i});
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
    
    
    
    
    combinedHist = [];
    
    %% loop through planes for colour images
    for j = 1 : size(img,3)
        %% SIFT_features = d*M where M is num of features sampled , d = 128.
        imgPlane = img(:,:,j);
        
        
        if(smoothing ==1)
            %% smooth before extracting features
            [imgPlane] = vl_imsmooth(imgPlane,sigma_smooth);
        end
        
        [locations, SIFT_features] = vl_dsift(single(imgPlane),'step',step,'fast', 'size',bin_size);
        
        
        %% Local clustering
        D = vl_alldist2(vocab,single(SIFT_features));
        
        
        [mindist,indices] = min(D,[],1);
        
        %% build histogram
        for k = 1 : size(mindist,2)
            imgHist(indices(k)) = imgHist(indices(k))+1;
        end
        
        %% normalise histogram
        imgHistNorm = imgHist./size(SIFT_features,2);
        
        %% add to combined histogram
        combinedHist = [combinedHist;imgHistNorm];
    end
    
    combinedHist = combinedHist';
    %add histogram to feature list
    image_feats = [image_feats;combinedHist];
end
end
