README 
Project 1, Machine Learning, EPFL 
Date: Oct 25, 2018 
Team members: Virginia Bordignon, Zilu Tian, Tatiana Volkova 

Please make sure "train.csv" and "test.csv" are in the same folder as run.py 

Detailed steps
1. Data preprocessing
	"load_clean_csv" is a helper function that build on load_csv. Additional arguments include missing_val and normalized. 
	missing_val specifies what to do for missing values, which are identified by -999: ignore, replace with the average of the column, or replace with the median. Average returns the best result. 
	normalized specifies whether to normalize the data to Gaussian, i.e. subtract the mean and divide by the standard deviation. As an improvement, we can also normalize data using other methods, which might better accommodate skewed distribution. However, we didn't further consider it in this project. 
2. Feature selection 
	"stepwise_regression" is a regression method that maintains a feature list and only populates the list when the rmse of the resulting feature list outperforms existing ones. 
	Alternatively, we also computed pairwise correlation and filtered the independent features, which can be seen in the Supplementary section of run.py. We chose to use stepwise_regression since it resulted in better features upon testing. 
3. Model selection 
	"build_poly_plus" builds a polynomial model of the given degree. It is coined with "plus" since it considers combinations of different features for a given degree as well. 
4. Regression method 
	"find_desired_val" returns the desired variable value of the given search space using grid search. It takes a function as an argument, which makes it easy to test for alternative regression methods. After comparing different methods, we decided to use ridge_regression since it achieves good performance. 
	"find_weight" applies cross validation by splitting data k_fold and  the final weight matrix is the average over matrices that result in least rmse for each run. 
5. Generate test result 
	"predict_labels" is a helper function provided
