# ml-certification
These are the projects I completed during my certification in machine learning last summer, and the specific functions I had to complete.

#### Week 2: Linear and Multivariate Linear Regression

In this project, I implement linear regression.
  - ***computeCost.m***: Cost function for linear regression
  - ***gradientDescent.m***: Gradient descent function
  - ***computeCostMulti.m***: Cost function for multivariate linear regression
  - ***gradientDescentMulti.m***: Gradient descent for multiple variables
  - ***featureNormalize.m***: Feature normalization
  - ***normalEqn.m***: Finding optimal parameters via the normal equation

#### Week 3: Logistic Regression

In this project, I implement logistic regression and apply it to two different datasets.
  - ***plotData.m***: Plotting 2D classification data
  - ***sigmoid.m***: Sigmoid function
  - ***costFunction.m***: Logistic regression cost function
  - ***predict.m***: Logstic regression prediction function
  - ***costFunctionReg.m***: Regularized logistic regression cost
  
#### Week 4: One vs. All Logistic Regression and Neural Networks

In this project, I implement one-vs-all logistic regression and neural networks to recognize handwritten digits
  - ***lrCostFunction.m***: Cost function for one-vs-all logistic regression
  - ***oneVsAll.m***: Training a one-vs-all multi-class classifier
  - ***predictOneVsAll.m***: Creating predictions using the one-vs-all classifier
  - ***predict.m***: Prediction function for neural network 

#### Week 5: Neural Network Learning (Backpropagation)

In this project, I implement backpropagation into the neural network from last week.
  - ***sigmoidGradient.m***: Computing the gradient of the sigmoid function
  - ***randInitializeWeights.m***: Randomly initializing the weights of each connection
  - ***nnCostFunction.m***: Cost function for neural network
  
#### Week 6: Regularized Linear Regression and Bias vs. Variance

In this project, I implement regularized linear regression and use it to study models using different bias-variance properties. 
  - ***linearRegCostFunction.m***: Cost function for regularized linear regression
  - ***learningCurve.m***: Learning curve generation
  - ***polyFeatures.m***: Map data into a polynomial feature space
  - ***validationCurve.m***: Cross validation curve generation

#### Week 7: Support Vector Machines (SVMs)

In this project, I use SVMs to classify emails as legitimate or spam.
  - ***gaussianKernel.m***: Gaussian kernel for SVM
  - ***dataSet3Params.m***: Parameters for the dataset
  - ***processEmail.m***: Email preprocessing
  - ***emailFeatures.m***: Email feature extraction
  
#### Week 8: K-means Clustering and Principal Component Analysis (PCA)

In this project, I use K-means clustering to compress an image. Afterwards, I use principal component analysis to find a low-dimensional representation of face images.
  - ***pca.m***: Perform principal component analysis 
  - ***projectData.m***: Project a dataset into a lower dimensional space
  - ***recoverData.m***: Recovers the original data from the projection
  - ***findClosestCentroids.m***: Finds the closest centroids for k-means
  - ***computeCentroids.m***: Computes the centroid means for k-means
  - ***kMeansInitCentroids.m***: Initializes the k-means centroids
 
#### Week 9: 

In this project, I implemented anomaly detection to detect failing servers on a network. Afterwards, I used collaborative filtering to build a recommender systems for movies.
  - ***estimateGaussian.m***: Estimate the parameters of a Gaussian distribution with a diagonal covariance matrix
  - ***selectThreshold.m***: Find a threshold for anomaly detection
  - ***cofiCostFunc.m***: Cost function for collaborative filtering
