function [ predicted_categories ] = nearest_neighbor_classify2( train_features, train_labels, test_features,k,categories,DISTANCE_TYPE)

%nearest_neighbor_classify Using k-nearest neighbor algorithm predicts the
%       scene category for each image
%   returns:    Nx1 cell array where each cell corresponds to a predicted
%               category and N is the number of test images
%
%   parameters:
%       k = number of nearest neighbors considered
%       distanceType = function to be passed to the pdsit2 interface
%       @histogram_intersection, @chi_square statistics


testDataLength = size(test_features,1);
[trainDataLength,featuresSize] = size(train_features);

predicted_categories = cell(testDataLength,1);
[kdists, kindices] = pdist2(train_features,test_features,DISTANCE_TYPE,'Smallest',k);

for i=1:trainDataLength
    % FOR EVERY TEST IMAGE DISTANCE VECTOR
    % each row in distanceMatrix corresponds to the distances between one
    % test image and all train images
    
        
    %  get category names from train image indices
    categoryNames = train_labels(kindices(:,i));
    
    %  get category indices from categoryNames
    categoryIndices = zeros(length(categoryNames),1);
    
    for n=1:length(categoryNames)
        categoryIndices(n) = find(strcmp(categories,categoryNames(n)));
    end
    
    if i==990
        disp('dsa');
    end
    %  get mode
    [isMode,modeValue] = getMode(categoryIndices);
    
    % if the mode does not exist - pick the first value (smallest distance)
    if isequal(isMode,false)
        predictedCatIndex = categoryIndices(1);
    else
        predictedCatIndex = modeValue;
    end
    
      
    predicted_categories(i,:) = categories(predictedCatIndex);
    
   
end







