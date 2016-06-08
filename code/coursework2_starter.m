%% Step 0: Set up parameters, vlfeat, category list, and image paths.


%% ---- CHOOSE FEATURE ---- %%
%FEATURE = 'tiny image';
%FEATURE = 'colour histogram';
FEATURE = 'bag of sift';
%FEATURE = 'bag of sift - michal';
michalVoc = 0; % 0- don't use 1 - use
%FEATURE = 'spatial pyramids';
%FEATURE = 'hogs';

%% ---- TINYIMAGE OPTIONS ----
DIMENSION_SIZE = 4;
METHOD = 'fit';
NORMALISE = 'unit-length';
COLOUR = 'rgb';

%% ---- COLOUR HISTOGRAM OPTIONS ----
QUANTISATION_LEVEL = 8;
COLOUR_SPACE = 'lab';

%% ---- SIFT VOCABULARY OPTIONS ----
vocab_size = 50;
bin_size = 3;
v_smooth_sigma = 1;
sift_smoothing = 1;
colour = 'greyscale';
%colour = 'rgb';
%colour = 'hsv';
%colour = 'w-sift';
v_step = 20;

%% ---- BAG OF SIFTS OPTIONS ----
s_smooth_sigma = 1;
s_step = 6;

%% ---- CHOOSE CLASSIFIER ----
CLASSIFIER = 'nearest neighbor';
%CLASSIFIER = 'support vector machine';

%% ---- kNN CLASSIFIER OPTIONS ----
K =5;
%DISTANCE_TYPE = @histogram_intersection;
DISTANCE_TYPE =@chi_square_statistics;
%DISTANCE_TYPE ='euclidean';
%DISTANCE_TYPE='cityblock'; 

%% ---- SVN CLASSIFIER OPTIONS ---
lambda = 10^-4;
run('vlfeat/toolbox/vl_setup')

%% ---- SPATIAL PYRAMID ---
sp_step = 8;
sp_size = 16;
pyramidLevel = 0; % 0 - 1 histogram, 1 - 5 histograms, 2 - 21

data_path = '../data/';

%% ---- HOG
%% VOCABULARY
hog_vocab_size = 50;
hog_colour = 'rgb';
smoothing = 0;
hog_smooth_sigma = 1;
hog_cell_size = 8;

%This is the list of categories / directories to use. The categories are
%somewhat sorted by similarity so that the confusion matrix looks more
%structured (indoor and then urban and then rural).
categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'House', ...
    'Industrial', 'Stadium', 'Underwater', 'TallBuilding', 'Street', ...
    'Highway', 'Field', 'Coast', 'Mountain', 'Forest'};

%This list of shortened category names is used later for visualization.
abbr_categories = {'Kit', 'Sto', 'Bed', 'Liv', 'Hou', 'Ind', 'Sta', ...
    'Und', 'Bld', 'Str', 'HW', 'Fld', 'Cst', 'Mnt', 'For'};

%number of training examples per category to use. Max is 100. For
%simplicity, we assume this is the number of test cases per category, as
%well.
num_train_per_cat = 100;

%This function returns cell arrays containing the file path for each train
%and test image, as well as cell arrays with the label of each train and
%test image. By default all four of these arrays will be 1500x1 where each
%entry is a char array (or string).
fprintf('Getting paths and labels for all train and test data\n')
[train_image_paths, test_image_paths, train_labels, test_labels] = ...
    get_image_paths(data_path, categories, num_train_per_cat);
%   train_image_paths  1500x1   cell
%   test_image_paths   1500x1   cell
%   train_labels       1500x1   cell
%   test_labels        1500x1   cell

%% Step 1: Represent each image with the appropriate feature
% Each function to construct features should return an N x d matrix, where
% N is the number of paths passed to the function and d is the
% dimensionality of each image representation. See the starter code for
% each function for more details.

fprintf('Using %s representation for images\n', FEATURE)

switch lower(FEATURE)
    case 'tiny image'
        if ~exist('image_feats_TI.mat', 'file')
        train_image_feats = get_tiny_images(train_image_paths,DIMENSION_SIZE,METHOD,NORMALISE,COLOUR);
        test_image_feats  = get_tiny_images(test_image_paths,DIMENSION_SIZE,METHOD,NORMALISE,COLOUR);
        save('image_feats_TI.mat', 'train_image_feats', 'test_image_feats')
        else
            load('image_feats_TI.mat');
        end
    case 'colour histogram'
        if ~exist('image_feats_CH.mat', 'file')
        train_image_feats = get_colour_histograms(train_image_paths,QUANTISATION_LEVEL,COLOUR_SPACE);
        test_image_feats  = get_colour_histograms(test_image_paths,QUANTISATION_LEVEL,COLOUR_SPACE);
        save('image_feats_CH.mat', 'train_image_feats', 'test_image_feats')
        else
            load('image_feats_CH.mat')
        end
    case 'bag of sift'
        
        if ~exist('vocab.mat', 'file')
            fprintf('No existing dictionary found. Computing one from training images\n')
            if(michalVoc ==1)
            vocab = build_vocabulary_MM(train_image_paths, vocab_size);
            
            else
                vocab = build_vocabulary(train_image_paths, vocab_size,colour,sift_smoothing,v_smooth_sigma,v_step,bin_size);
            end
            save('vocab.mat', 'vocab')
        end
        
        if ~exist('image_feats.mat', 'file')
            fprintf('No existing feats found. Computing...\n')
            train_image_feats = get_bags_of_sifts(train_image_paths,vocab_size, colour,sift_smoothing, s_smooth_sigma, s_step, bin_size);
            test_image_feats  = get_bags_of_sifts(test_image_paths,vocab_size, colour,sift_smoothing, s_smooth_sigma, s_step, bin_size);
            save('image_feats.mat', 'train_image_feats', 'test_image_feats')
        else
            load('image_feats.mat')
        end
        
        case 'bag of sift - michal'
        
        if ~exist('vocab.mat', 'file')
            fprintf('No existing dictionary found. Computing one from training images\n')
            if(michalVoc ==1)
            vocab = build_vocabulary_MM(train_image_paths, vocab_size);
            
            else
                vocab = build_vocabulary(train_image_paths, vocab_size,colour,v_smooth_sigma,v_step,bin_size);
            end
            save('vocab.mat', 'vocab')
        end
        
        if ~exist('image_feats.mat', 'file')
            fprintf('No existing feats found. Computing...\n')
            train_image_feats = get_bags_of_sifts_MM(train_image_paths);
            test_image_feats  = get_bags_of_sifts_MM(test_image_paths);
            save('image_feats.mat', 'train_image_feats', 'test_image_feats')
        else
            load('image_feats.mat')
        end
    case 'spatial pyramids'
         if ~exist('vocab.mat', 'file')
            fprintf('No existing dictionary found. Computing one from training images\n')
            vocab = build_vocabulary(train_image_paths, vocab_size,colour,smoothing,v_smooth_sigma,v_step,bin_size);
            save('vocab.mat', 'vocab')
         end
        
          if ~exist('image_feats.mat', 'file')
            fprintf('No existing feats found. Computing...\n');
            train_image_feats = get_spatial_pyramids(train_image_paths,vocab_size,s_smooth_sigma,sp_step,sp_size,pyramidLevel);
            test_image_feats  = get_spatial_pyramids(test_image_paths,vocab_size, s_smooth_sigma, sp_step, sp_size, pyramidLevel);
            save('image_feats.mat', 'train_image_feats', 'test_image_feats')
          else
            load('image_feats.mat')
          end
          
    case 'hogs'
        if ~exist('vocab.mat', 'file')
            fprintf('No existing dictionary found. Computing one from training images\n')
            vocab = build_vocabularyHOG(train_image_paths, hog_vocab_size,hog_colour,smoothing,hog_smooth_sigma,hog_cell_size);
            save('vocab.mat', 'vocab')
         end
        
          if ~exist('image_feats.mat', 'file')
            fprintf('No existing feats found. Computing...\n');
            train_image_feats = get_hogs(train_image_paths, hog_vocab_size,hog_colour,smoothing,hog_smooth_sigma,hog_cell_size);
            test_image_feats  = get_hogs(test_image_paths, hog_vocab_size,hog_colour,smoothing,hog_smooth_sigma,hog_cell_size);
            save('image_feats.mat', 'train_image_feats', 'test_image_feats')
          else
            load('image_feats.mat')
          end
        
end
%% Step 2: Classify each test image by training and using the appropriate classifier
% Each function to classify test features will return an N x 1 cell array,
% where N is the number of test cases and each entry is a string indicating
% the predicted category for each test image. Each entry in
% 'predicted_categories' must be one of the 15 strings in 'categories',
% 'train_labels', and 'test_labels'. See the starter code for each function
% for more details.

fprintf('Using %s classifier to predict test set categories\n', CLASSIFIER)

switch lower(CLASSIFIER)
    case 'nearest neighbor'
        predicted_categories = nearest_neighbor_classify2(train_image_feats, train_labels, test_image_feats,K,categories,DISTANCE_TYPE);
    case 'support vector machine'
        predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats, lambda);
    case 'nearest neighbor p'
        predicted_categories = nearest_neighbor_classify_old(train_image_feats, train_labels, test_image_feats);
end

%% Step 3: Build a confusion matrix and score the recognition system
% You do not need to code anything in this section.

% This function will recreate results_webpage/index.html and various image
% thumbnails each time it is called. View the webpage to help interpret
% your classifier performance. Where is it making mistakes? Are the
% confusions reasonable?
create_results_webpage( train_image_paths, ...
    test_image_paths, ...
    train_labels, ...
    test_labels, ...
    categories, ...
    abbr_categories, ...
    predicted_categories)