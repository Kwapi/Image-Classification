function image_feats = get_spatial_pyramids(image_paths,vocab, smooth, step,bin_size,pyrlevel)
%colour defines whether using sift with colour or grayscale
%step defines the step size for sift

% image_paths is an N x 1 cell array of strings where each string is an
% image path on the file system.

% This function assumes that 'vocab.mat' exists and contains an N x 128
% matrix 'vocab' where each row is a kmeans centroid or visual word. This
% matrix is saved to disk rather than passed in a parameter to avoid
% recomputing the vocabulary every time at significant expense.

% image_feats is an N x d matrix, where d is the dimensionality of the
% feature representation. In this case, d will equal the number of clusters
% or equivalently the number of entries in each image's histogram.

%{
Useful functions:
[locations, SIFT_features] = vl_dsift(img) 
 http://www.vlfeat.org/matlab/vl_dsift.html
 locations is a 2 x n list list of locations, which can be used for extra
  credit if you are constructing a "spatial pyramid".
 SIFT_features is a 128 x N matrix of SIFT features
  note: there are step, bin size, and smoothing parameters you can
  manipulate for vl_dsift(). We recommend debugging with the 'fast'
  parameter. This approximate version of SIFT is about 20 times faster to
  compute. Also, be sure not to use the default value of step size. It will
  be very slow and you'll see relatively little performance gain from
  extremely dense sampling. You are welcome to use your own SIFT feature
  code! It will probably be slower, though.

D = vl_alldist2(X,Y) 
   http://www.vlfeat.org/matlab/vl_alldist2.html
    returns the pairwise distance matrix D of the columns of X and Y. 
    D(i,j) = sum (X(:,i) - Y(:,j)).^2
    Note that vl_feat represents points as columns vs this code (and Matlab
    in general) represents points as rows. So you probably want to use the
    transpose operator '  You can use this to figure out the closest
    cluster center for every SIFT feature. You could easily code this
    yourself, but vl_alldist2 tends to be much faster.
%}

load('vocab.mat')

imageNum = length(image_paths);

%0 level histogram
imgHist0 = zeros(size(vocab,2),1);
image_feats = [];%zeros(imageNum,d);

%% Read in images and construct sift histograms
for i =1 :imageNum 

    img = imread(image_paths{i});
     img = single(img);
    img = rgb2gray(img);
   
    [size_y,size_x] = size(img);
    noPixels = size_y * size_x;
    step = 3;
    
    %create array to store pyramid histograms
    
    
    %SIFT_features = d*M where M is num of features sampled , d = 128.
    %locations 2*n list of locations
    
    %manipulate step, bin size, and smoothing parameters
    [locations, SIFT_features] = vl_dsift(img,'step',step,'size',bin_size,'fast');
 
    %convert matrix to single precision
    SIFT_features = single(SIFT_features);
        
    %workout local cluster
    D = vl_alldist2(vocab,SIFT_features);
       
    %assign local feature to nearest cluster center
    %min of each row of distances corresponds to closest
    [mindist,ind] = min(D);
            
    %build histograms
    noHistograms = 0;
    for z = 0 : pyrlevel
        noHistograms = noHistograms + 2^(2*z);
    end
    
    imgHist = zeros(size(vocab,2),noHistograms);
    
    %level 0          
    for j =1 : size(mindist,2)
          imgHist0(ind(j)) = imgHist0(ind(j))+1;
    end
    
    % register first histogram for level 0
    imgHist(:,1) = imgHist0;
    
    counter = 2; % 2 is the first level 1 histQuad index          
    %level >0
    %% LEVELS
    for k =1:pyrlevel
        %fprintf('level: %d\n', k);
          %how many quadrants at this level
          quadrantNum = 2^(2*k);
          % each quadrant has its own histogram
          
          %% find size of each cell
          quadSizeX = floor(size_x/quadrantNum) ;
          quadSizeY = floor(size_y/quadrantNum);
          
          %% QUADRANTS
          for m = 1 : quadrantNum
              %fprintf('quadrant : %d/%d\n',m,quadrantNum);
              histQuad = zeros(size(vocab,2),1);
              
              %% loop through the whole quadrant and get histogram - could probably be vectorised
              for x = (m-1)*quadSizeX + 1 : m * quadSizeX
                  for y = (m-1)*quadSizeY + 1 : m*quadSizeY
                                                    
                      %% check if there is a feature at this location
                      
                      %% check x
                      locX = locations(1,:) == x;
                      %% check y
                      locY = locations(2,:) == y;
                      
                      %% check if there is an overlap
                      locXY = bitand(locX,locY);
                      
                      locXYIndex = find(locXY);
                      
                      if(~isempty(locXYIndex))
                          %% we've got a feature
                          vocBin = ind(locXYIndex);
                          
                          %% add to quadrant histogram
                          histQuad(vocBin) = histQuad(vocBin) + 1;
                      end
                  end
              end
              
              imgHist(:,counter) = histQuad;              
              
             counter = counter + 1;
          end
       
    end
    
    % normalise histogram by number of pixels
    imgHist = imgHist/noPixels;
    
    
    % apply weighting
    %... apply weightings based  (1/2^(L - l) L is total level(pyrlevel) and  l
    %is what level were on (k)
    pointer = 1;
    for pL = 0 : pyrlevel
        q = 2^(2*pL);
        
        weight = (1/2)^(pyrlevel - pL); 
        
        if(pL ~= 0)
        startIndex = pointer;
        endIndex = pointer + q - 1;
        imgHist(:,startIndex:endIndex) = imgHist(:,startIndex:endIndex) * weight;
        else
            imgHist(:,1) = imgHist(:,1) * weight;
        end
        
        pointer = pointer + q;
    end
    
    
       
        
    %flatten to a 1d histogram
    imgHistFlat = imgHist(:);
    
    
    
    %addd histogram to feature list
    
    image_feats = [image_feats,imgHistFlat];
end
image_feats = image_feats';

end