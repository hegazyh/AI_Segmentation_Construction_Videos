%% Load the Annotated Data
filePath = 'D:\Infrastructure Projects\interlock flooring\annotations_InterlockFlooring.mat';
loadedData = load(filePath);

% Use the correct variable name (gTruth)
if isfield(loadedData, 'gTruth')
    groundTruthData = loadedData.gTruth;
else
    error('Variable "gTruth" not found in the loaded file.');
end

% Validate the groundTruth object
if ~isa(groundTruthData, 'groundTruth')
    error('The variable is not a valid groundTruth object.');
end

% Convert groundTruth object to a table for training
trainingData = objectDetectorTrainingData(groundTruthData);

% Check the variable names in trainingData
disp("Training data variable names:");
disp(trainingData.Properties.VariableNames);

%% Split the Data: 70% Training, 20% Validation, 10% Testing
totalData = height(trainingData);
numTrain = round(0.7 * totalData);
numVal = round(0.2 * totalData);

% Randomize the data order
shuffledData = trainingData(randperm(totalData), :);

% Split the data
trainData = shuffledData(1:numTrain, :);
valData = shuffledData(numTrain+1:numTrain+numVal, :);
testData = shuffledData(numTrain+numVal+1:end, :);

% Define the input size
inputSize = [416 416 3];

% Specify class names
classes = ["InterlockFlooring"];

%% Estimate Anchor Boxes
boxDatastore = boxLabelDatastore(trainData(:, {'InterlockFlooring'}));
numAnchors = 9; % Number of anchor boxes
anchorBoxes = estimateAnchorBoxes(boxDatastore, numAnchors);

% Rescale anchor boxes to fit within the input size
anchorBoxes = round(anchorBoxes .* (inputSize(1:2) ./ max(anchorBoxes(:))));

% Verify anchor boxes
disp("Final Anchor Boxes:");
disp(anchorBoxes);

%% Define the YOLO v2 Network
net = resnet50;
featureLayer = 'activation_40_relu'; % Feature extraction layer
lgraph = yolov2Layers(inputSize, numel(classes), anchorBoxes, net, featureLayer);

%% Normalize Input Images with Error Handling
normalizeImage = @(x) handleCorruptedImage(x);

function img = handleCorruptedImage(filename)
    try
        img = im2double(imread(filename)); % Read and normalize image
    catch
        warning(['Skipping corrupted file: ', filename]);
        img = []; % Return an empty array for corrupted images
    end
end

% Create imageDatastores for training and validation
imdsTrain = imageDatastore(trainData.imageFilename, 'ReadFcn', normalizeImage);
imdsVal = imageDatastore(valData.imageFilename, 'ReadFcn', normalizeImage);

% Remove invalid files
validTrainFiles = cellfun(@(x) ~isempty(handleCorruptedImage(x)), trainData.imageFilename);
validValFiles = cellfun(@(x) ~isempty(handleCorruptedImage(x)), valData.imageFilename);

trainData = trainData(validTrainFiles, :);
valData = valData(validValFiles, :);

% Update datastores after filtering
imdsTrain = imageDatastore(trainData.imageFilename, 'ReadFcn', normalizeImage);
imdsVal = imageDatastore(valData.imageFilename, 'ReadFcn', normalizeImage);

bldsTrain = boxLabelDatastore(trainData(:, {'InterlockFlooring'}));
bldsVal = boxLabelDatastore(valData(:, {'InterlockFlooring'}));

trainDatastore = combine(imdsTrain, bldsTrain);
validationDatastore = combine(imdsVal, bldsVal);

%% Set Training Options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 60, ...
    'MiniBatchSize', 8, ...
    'Shuffle', 'every-epoch', ...
    'GradientThreshold', 10, ...
    'ValidationData', validationDatastore, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% Train the YOLO v2 Detector
[detector, info] = trainYOLOv2ObjectDetector(trainDatastore, lgraph, options);

%% Evaluate the Model on Validation Data
results = detect(detector, validationDatastore, 'Threshold', 0.5);

% Evaluate performance using mAP
[ap, recall, precision] = evaluateDetectionPrecision(results, validationDatastore);
disp("Mean Average Precision (mAP): " + ap);

% Save the trained model
save('D:\Infrastructure Projects\interlock flooring\InterlockFlooringDetector.mat', 'detector');

%% Real-Time Detection with Videos
% Load the trained detector
load('D:\Infrastructure Projects\interlock flooring\InterlockFlooringDetector.mat', 'detector');

% Specify the input video file
videoFile = 'D:\Infrastructure Projects\interlock flooring\Test_3.mp4';
videoReader = VideoReader(videoFile);

% Prepare the output video file
outputVideo = VideoWriter('D:\Infrastructure Projects\interlock flooring\annotated_video.mp4', 'MPEG-4');
open(outputVideo);

% Process each frame in the video
disp("Processing video for real-time detection...");
while hasFrame(videoReader)
    % Read the next frame
    frame = readFrame(videoReader);
    
    % Run detection on the frame
    [bboxes, scores, labels] = detect(detector, frame, 'Threshold', 0.2);
    
    % Annotate the frame with detected objects
    if ~isempty(bboxes)
        annotatedFrame = insertObjectAnnotation(frame, 'rectangle', bboxes, labels);
    else
        annotatedFrame = frame; % Use the original frame if no objects are detected
    end
    
    % Write the annotated frame to the output video
    writeVideo(outputVideo, annotatedFrame);
    
    % Display the annotated frame (optional)
    imshow(annotatedFrame);
    pause(0.01); % Pause for a short time to simulate real-time playback
end

% Close the output video file
close(outputVideo);

disp("Annotated video saved as 'annotated_video.mp4'.");
