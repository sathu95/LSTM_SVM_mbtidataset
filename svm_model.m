%%This is a SVM model, where we train the text classifier on word of counts
%%using bag-of words. It is a model to predict the type of personality
%%using the text data on posts column

%%  %loading the  data from the final preprocessed csv
 text_clean = readtable('finaloutput.csv','TextType','string');
 head(text_clean)
 
%% converting the labels in the type column of the table to categorical and view the distribution of the classes in the data using a histogram
text_clean.type = categorical(text_clean.type);

 %histogram to see the distribution of the classes in the data 
f = figure;
f.Position(3) = 1.5*f.Position(3);

h = histogram(text_clean.type);
xlabel("Class")
ylabel("Frequency")
title("Class Distribution")


%% partitioning the data into the training and a held out test set. Specifying the holdout percentage to be 10%
cvp = cvpartition(text_clean.type,'Holdout',0.1);
dataTrain = text_clean(training(cvp),:);
dataTest = text_clean(test(cvp),:);


%% extracting the text data and labels from the partitioned tables
textDataTrain = dataTrain.posts;
textDataTest = dataTest.posts;
YTrain = dataTrain.type;
YTest = dataTest.type;

%%As all the preprocessing is done earlier, we now just tokenize the text
%%documents
documentsTrain = tokenizedDocument(textDataTrain);
documentsTrain(1:5)

%We create bag-of words model from the tokenized documents
bag = bagOfWords(documentsTrain);

%Removing words from the bag-of-words that do not appear more than four
%times in total(considering the size of the document) 
%Removing any documents containging no words from the bag-of-model, and
%remove the corresponding entries in labels
bag = removeInfrequentWords(bag,4);
[bag,idx] = removeEmptyDocuments(bag);
YTrain(idx) = [];

%% Training a multiclass linear classification model using fitcecoc. Specifying the Counts property of the bag-of-words model to be the predictors, and the event type labels to be the response. Specifying the learners to be linear.

XTrain = bag.Counts;
mdl = fitcecoc(XTrain,YTrain,'Learners','linear');

%% Here we predict the labels of the test data using the trained model.
%Similar preprocessing is done on the test data as well.

%tokenizing the documents 
documentsTest = tokenizedDocument(textDataTest);
documentsTest(1:5)

%This encodes the resulting test documents as a matrix of word frequency counts according to the bag-of-words model.
XTest = encode(bag,documentsTest);

%Predicting the labels of the test data using the trained model and calculating the classification accuracy.
YPred = predict(mdl,XTest);
acc = sum(YPred == YTest)/numel(YTest)

%plotting a confusion matrix for the test data
plotconfusion(YTest,YPred)

%Predicting type using latest Trump tweet and the result shows type as INFP

str = [ ...
    "I have agreed with the historically cooperative, disciplined approach that we have engaged in with Robert Mueller (Unlike the Clintons!). I have full confidence in Ty Cobb, my Special Counsel, and have been fully advised throughout each phase of this process."];

DocumentsNew = tokenizedDocument(str);
XNew = encode(bag,DocumentsNew);
labelsNew = predict(mdl,XNew)



