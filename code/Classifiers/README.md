Support Vector Machine (SVM)
SVM is a discriminative classifier formally defined by a separating hyperplane. 
In other words, given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples.

K nearest neighbors (KNN)
KNN is a simple algorithm that stores all available cases and classifies new cases based on a similarity measure.
The training examples are vectors in a multidimensional feature space, each with a class label.
The training phase of the algorithm consists only of storing the feature vectors and class labels of the training samples.
In the classification phase, k is a user-defined constant, and an unlabeled vector (a query or test point) is classified by assigning the label which is most frequent among the k training samples nearest to that query point.

extreme Gradient Boosting (XGBoost)
XGBoost library implements the gradient boosting decision tree algorithm.
This algorithm goes by lots of different names such as gradient boosting, multiple additive regression trees, stochastic gradient boosting or gradient boosting machines. 
Gradient boosting is an approach where new models are created that predict the residuals or errors of prior models and then added together to make the final prediction.
It is called gradient boosting because it uses a gradient descent algorithm to minimize the loss when adding new models.
