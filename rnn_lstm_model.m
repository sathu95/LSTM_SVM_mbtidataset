%%This model is built to classify text tweets using deep learning Long
%%short term memory (LSTM) network
%We have used LSTM to learn and use the long term dependencies to classify
%sequence data



%% %loading the  data 
%Importing mbti data. This data contains labeled textual descriptions of individual posts. To import the text data as string arrays, we specify the text type to be 'string'.
 text_clean = readtable('finaloutput.csv','TextType','string');
 head(text_clean)


 %% Converting the labels in the type column to the table to categorical
text_clean.type = categorical(text_clean.type);
 
 %%histogram to see the distribution of the classes in the data 
f = figure;
f.Position(3) = 1.5*f.Position(3);

h = histogram(text_clean.type);
xlabel("Class")
ylabel("Frequency")
title("Class Distribution")


%%Partitioning the data into the training and a held out test set. Specifying the holdout percentage to be 10%
cvp = cvpartition(text_clean.type,'Holdout',0.1);
dataTrain = text_clean(training(cvp),:);
dataTest = text_clean(test(cvp),:);


%% Extracting the text data and labels from the partitioned tables
textDataTrain = dataTrain.posts;
textDataTest = dataTest.posts;
YTrain = dataTrain.type;
YTest = dataTest.type;


%%Vizualizing the trainign text data using the word cloud
figure
wordcloud(textDataTrain);
title("Training Data")

%%As all the preprocessing is done earlier, we now tokenize the text
documentsTrain = tokenizedDocument(textDataTrain);
documentsTrain(1:5)


%% Mapping words in a vocubulary to numeric vectors. Training a word embedding with dimension 100. And the training epochs are set at 50, Verbose output is set to 0 to suppress the output. 
%%Takes several minutes%%
embeddingDimension = 100;
embeddingEpochs = 50;

emb = trainWordEmbedding(documentsTrain, ...
    'Dimension',embeddingDimension, ...
    'NumEpochs',embeddingEpochs, ...
    'Verbose',0)

words = emb.Vocabulary;
words(1:500)

%% Plotting histogram to check the length of the documents

documentLengths = doclength(documentsTrain);

figure 
histogram(documentLengths)
title("Document Length")
xlabel("Length")
ylabel("Number of Documents")


%% %Most of the documents have tokens between 1200 to 1700
%Truncate the training document to have the length of 1700 using docfun.
%This function inputted to docfun takes the string array input and output
%the first 1700 elements.

sequenceLength = 1700;
documentsTruncatedTrain = docfun(@(words) words(1:min(sequenceLength,end)),documentsTrain);

%Converting the documents to sequences of word vectors
XTrain = doc2sequence(emb,documentsTruncatedTrain);
XTrain(1:5)





%% LSTM Network architecture %%

%This part of the code includes a sequence input layer, which is set to the
%size of word embedding. Next,layer that specify the output size which is
%set to 200. To use the LSTM layer for a sequence-to-label classification problem,output mode is set to 'last'.
%Finally, a fully connected layer with the same size as the number of
%classes, a softmax layer and a classification layer.

inputSize = embeddingDimension;
outputSize = 200;
numClasses = numel(categories(YTrain));

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(outputSize,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]

%% Specifying training options
%The solver is set to 'adam', and gradiant threshold is set to 1. To
%monitor the training process, we have set the 'plots' option to
%'training-progress'. Other options such as batch size, epochs and learning
%rate modifers are provided to adjust them to get the better results.

options = trainingOptions('adam',...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',5,...
    'ExecutionEnvironment', 'cpu',...
    'MaxEpochs',140,...
    'MiniBatchSize',128,...
    'Plots','training-progress')

%% Training the LSTM network using the trainNetwork function
net = trainNetwork(XTrain,YTrain,layers,options)

[XTrain,YTrain] = digitTrain4DArrayData;

idx = randperm(size(XTrain,4),1000);
XValidation = XTrain(:,:,:,idx);
XTrain(:,:,:,idx) = [];
YValidation = YTrain(idx);
YTrain(idx) = [];

%% Testing data using same steps that were used as the training documents

%Tokenizing the test set
documentsTest = tokenizedDocument(textDataTest);

%Convert the test documents to sequences using the same steps as the training documents.


documentsTruncatedTest = docfun(@(words) words(1:min(sequenceLength,end)),documentsTest);
XTest = doc2sequence(emb,documentsTruncatedTest);
for i=1:numel(XTest)
    XTest{i} = leftPad(XTest{i},sequenceLength);
end
XTest(1:5)


%Classifying the test documents using the trained LSTM network.

YPred = classify(net,XTest);

%Calculating the classification accuracy. The accuracy is the proportion of labels that the network predicts correctly.

accuracy = sum(YPred == YTest)/numel(YPred)

%% Predicting type of a tweet which was unseen before
%All the steps before we try to pridect the type are similar to that of the
%training and testing steps (Till converting text data into sequences and
%padding the length of the document.


textNew = [ ...
    "I have agreed with the historically cooperative, disciplined approach that we have engaged in with Robert Mueller (Unlike the Clintons!). I have full confidence in Ty Cobb, my Special Counsel, and have been fully advised throughout each phase of this process."];

documentsNew = tokenizedDocument(textNew);

documentsTruncatedNew = docfun(@(words) words(1:min(sequenceLength,end)),documentsNew);
XNew = doc2sequence(emb,documentsTruncatedNew);
for i=1:numel(XNew)
    XNew{i} = leftPad(XNew{i},sequenceLength);
end

%Classify the new sequences using the trained LSTM network.
[labelsNew,score] = classify(net,XNew);

%Showing the output with their predicted labels.
[textNew string(labelsNew)]

%To view the top three predictions and their scores for the first report.
[scoreTop,idxTop] = maxk(score(1,:),3);

% For getting the class names from the classification output layer (the last layer) of the LSTM network.
classNames = net.Layers(end).ClassNames;



%% Function to convert documents to sequence of word vectors and pad the document length

%Converting the documents to sequences of word vectors

function C = doc2sequence(emb,documents)

parfor i = 1:numel(documents)
    words = string(documents(i));
    idx = ~ismember(emb,words);
    words(idx) = [];
    C{i} = word2vec(emb,words)';
end

end

%The function leftPad pads matrix M with zeros on the left so that it has N columns. 
%As per theory right is automatically padded


function MPadded = leftPad(M,N)

[dimension,sequenceLength] = size(M);
paddingLength = N-sequenceLength;
MPadded = [zeros(dimension,paddingLength) M];

end
