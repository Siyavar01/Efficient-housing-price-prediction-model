# Efficient-housing-price-prediction-model-
An efficient housing price prediction model using advanced linear regression techniques and Ensemble model

Abstract
As in the current scenario, the real estate prices fluctuate a lot and it’s very difficult for a naïve person to navigate through these prices. This creates a need for a prediction model which bridges the gap between the problem and solution for the community. There are pre-existing approaches in the literature to fulfil this demand. But these approaches suffer from overfitting issues and are not generalized for all the type of datasets. To overcome such issues this paper proposes a new approach by adopting advanced linear regression techniques such as kernel ridge, lasso and elastic net and ensemble decision trees algorithm like LightGBM, gradient boosting, and XGBoost regression models. These approaches have been tested on the dataset derived from Kaggle. This dataset consists of 81 features and 1460 instances. The results indicate the superiority of the adopted approaches over the pre-existing literature. To further explore the best approach amongst the adopted approaches, results have been collected and lasso approach is found to be superior. At the end, we have used Stacking which is an ensemble machine learning algorithm and for that lasso is selected as the meta-learning algorithm as it performs the best among the proposed approaches. Therefore, our proposed model gives very accurate results when compared to the pre-existing techniques provided by fellow researchers. The ensemble model produces an accuracy of approximately 99.6% and a RMSE value of 0.0682. Hence, our proposed model indeed performs substantially better and therefore could very well be used to predict housing prices accurately.

Introduction
As the saying goes, housing is one of the most primal needs of a person and every one of us strives to get the best deal possible when it comes to real estate prices. Since the ancient times, the mankind has been dependent upon the real estate brokers, stakeholders, real estate dealers, financial institutions and market predictors for seeking advice about the real estate market. The human mind has endless capabilities but in some cases its decisions are prone to error and biasness. Driven by these errors of the decision makers and the false advertisements which are constantly targeted at them, buyers end up in a mess and eventually incur substantial losses. The predictions have been based on field work and experience rather than data and models. Due to the scarcity of data and innovation, both the businesses and individuals had suffered in the past. This creates a need for a strong prediction model which bridges the gap between the buyers and their need for accurate housing price predictions. Throughout these years we’ve seen tremendous advancements in the field of computer science consequence of which big data has emerged to be one of the front runners in the trends corresponding to the domain. This has resulted into machine learning becoming a vital approach when it comes to prediction of housing prices as it does so very efficiently. With the evolution of Artificial Intelligence (A.I.) and Machine learning (M.L.), numerous researchers have proposed their own models to solve this problem.

Truong et al. 2020 proposed Random Forest, XGBoost, LightGBM, Hybrid Regression and Stacked generalization models. The model which performed the best on the training set came out to be Hybrid regression with RMSE value as 0.14969 [1]. Pai et al. 2020 proposed least squares support vector regression (LSSVR), classification and regression tree (CART), general regression neural networks (GRNN), and backpropagation neural networks (BPNN) models. The least squares support vector regression performed the best (with MAPE - 0.228 % and NMAE  - 8.11 * 10^-4)  among the following models [2]. Modi et al. 2020 proposed Logistic Regression, Support Vector Machine, Extra Tree, Naïve Bayes and Ensemble learning techniques such as bagging, boosting, and stacking. The meta classifier that is used is the voting classifier which performs soft voting to get the desired outcome gets the best results with an accuracy of 98.27% and precision of 97.98% [3].

These cited approaches don’t pass the test of time very well. For instance, Linear Regression and Decision Tree Regression have been applied since the old ages and the evolution of ML has brought in numerous effective models into the picture. These said approaches suffer from the issue of overfitting and hence affects the overall accuracy of price prediction. The attempt is to overcome the issues that are there in the models which are currently deployed and come up with a model which performs much better in comparison to the existing literature. In this paper, we’ve proposed a model by adopting advanced linear regression techniques such as kernel ridge, lasso and elastic net and ensemble decision trees algorithm like LightGBM, gradient boosting, and XGBoost regression models. This model involves the implementation of an ensemble learning method i.e., Stacking to provide more accurate and precise outcome. These approaches have been tested on the dataset derived from Kaggle. This dataset consists of 81 features and 1460 instances. The rest of this paper is organized as follows. Section 2 includes Literature Review. Section 3 includes Research Methodology. Section 4 includes Results. Section 5 includes Conclusion. Section 6 includes References.
Literature Review
Literature review forms the heart of any research paper and while going through the pre-existing literature concerning our domain, we surely left no stone unturned. After analysing and studying 35 said research papers, we compiled a list of these 20 papers related to our field of interest. The following table forms the backbone of our existing literature as it contains the concise description of all the papers which make use of datasets similar to the one which we’ve derived from Kaggle.

Author	Approaches/Models used	Remarks
Yu et al. 2016 [4]
Gaussian Naïve Bayes, Multinomial Naive Bayes, SVC with linear kernel and SVR with gaussian kernel	SVR with gaussian kernel performed the best with RMSE of 0.5271
Mullainathan et al. 2017[5]
Lasso Regression, Random Forest Regression, Ensemble and Regression tree tuned by depth	Random Forest performed the best with an accuracy of 85.4% on the training data
Manjula et al. 2017[6]
Simple Linear Regression, Multivariate Regression models, Ridge Regression and Lasso Regression	A need to create a mix or an ensemble of these models is stated to increase the accuracy
Banerjee et al. 2018[7]
Support Vector Machines, Random Forest, Artificial Neural Network	Support Vector Machine classifier can be said to be reliable with an accuracy of 82%
Ghosalkar et al. 2018[8]
Linear Regression	Linear Regression gives out the minimum prediction error of 0.3713
Varma et al. 2018[9]
Linear Regression, Boosted Regression, Forest Regression and Neural Networks	The weighted mean of the given techniques were taken into consideration to give most accurate results
Phan 2019[10]
Linear Regression, Polynomial Regression, Principal Component Analysis, Regression Tree, Neural Network and Support Vector Machine	Tuned Support Vector Machine performed the best with the evaluation ratio of 0.56
Truong et al. 2020[1]
Random Forest, XGBoost, LightGBM, Hybrid Regression and Stacked generalization	Hybrid regression performed the best with RMSE value 0.14969
Pérez-Rave et al. 2020[11]
Linear regression, Regression trees, Random Forest, Bagging
	Bagging performed the best with R^2 value 99.4%
Jha et al. 2020[12]
Logistic Regression, Random Forest, Voting Classifier, and XGBoost	XGBoost delivers superior results
Al-Gbury et al. 2020[13]
Artificial neural network and a grey wolf optimizer	ANN (with 6 neurons) performs better with an accuracy of 98.7% 
Pai et al. 2020[2]
Least squares support vector regression (LSSVR), classification and regression tree (CART), general regression neural networks (GRNN), and backpropagation neural networks (BPNN)	The least squares support vector regression performed the best (with MAPE - 0.228 % and NMAE - 8.11 * 10^-4) among the following models
Modi et al. 2020[3]
Logistic Regression, Support Vector Machine, Extra Tree, Naïve Bayes and Ensemble learning techniques such as bagging, boosting, and stacking	The meta classifier that is used is the voting classifier which performs soft voting to get the desired outcome gets the best results with an accuracy of 98.27% and precision of 97.98%
Kang et al. 2020[14]
Fuzzy mathematics is used to optimize the typical multiple regression model	The fuzzy set theory has been applied to the data analysis algorithm, which has improved the scientificity and effectiveness of the algorithm
Kang et al. 2020[15]
	Multiple linear regression (MLR) and gradient boosting machine (GBM)
	Gradient boosting machine (GBM) performed the best with R2 = 0.74; RMSE = 0.077

Uzut et al. 2020[16]
Random forest, gradient boosting and linear regressor	The best result was obtained by gradient boosting regression when the
amount of test set is 20%, and the mean absolute error for this method was 3.92
Chauhan et al. 2021[17]
Bootstrap Aggregating, Random Forest, Adaptive Boosting and Gradient Boosting	The gradient boosting algorithm worked pretty well with a satisfactory accuracy score and lesser MAPE ie. 18.67%.
Chaturvedi et al. 2021[18]
Supplying regression, support vector regression, Lasso Regression technique and call Tree	Call tree performed the best with RMSE value 0.99
Tripathi 2021[19]
Random Forest, Multiple Regression, Support Vector Machine, Gradient Boosting, Neural Networks, Ensemble learning Bagging	Random Forest performs the best with Accuracy - 90% and RMSE - 0.012
Kangane et al. 2021[20]
Multiple Linear Regression, Linear SVM, Decision Tree, Ridge, Lasso, Gradient Boosting, XGBoosting, CATBoost and Random Forest Regression	XGBoost Regression performs the best with highest model score of 93.14%

After analysing this given table, it can be clearly indicated that there is a subsequent scope of improvement in the accuracy of the predictions. Like we’ve seen in the remarks section of Mullainathan et al. 2017[5], Banerjee et al. 2018[7] and Tripathi 2021[19]. This creates a need to bridge these errors with a better prediction model which in turn help the industry and the buyers who are in the real estate market.

Methodology
The research methodology lies at the core of our research paper implementation. It can be summarized with the help of flow chart given below. The complete work process can be divided into 5 segments. These are: Introduction to the Dataset, Data Visualization, Data Pre-processing, Model deployment and Evaluating model's performance.

 


	Introduction to the Dataset
The dataset has been derived from Kaggle which is an online application. This dataset is part of an ongoing competition being currently held on the given site. The dataset has 1460 instances and 81 attributes out of which SalePrice is our target variable. The dataset comprises of two .csv files i.e., train.csv and test.csv. Out of the given 81 attributes, 10 attributes are picked out which are of more importance for the analysis and their description is given below:

	Data Visualization
Data Visualization is very important when it comes to machine learning and data science implementation. Data when being read in the form of graphs and charts, it becomes easier to comprehend and derive conclusions from it. Thus, it holds the key for getting accurate results. Python provides us with various visualization tools like matplotlib and seaborn. We’ve made use of numerous graphs like joint plots and box plots to establish key relationships between the attributes and moving on it’ll also help in data pre-processing.
The first visualization tool that we’re going to be using is Correlation matrix. It is provided under the sklearn library. It’s simply a matrix denoting the linear relationship between the given attributes. 
The Correlation Matrix Heatmap is given as:
 
                  Figure I
From this, we can come up with the 10 most correlated features for our dataset and they’re given as follows:

Attribute Name	Description	Data type
SalePrice	The property's sale price in dollars. This is the target variable that we're trying to predict.	Integer
OverallQual	Overall material and finish quality	Integer
GrLivArea	Above grade (ground) living area square feet	Integer
GarageCars	Size of garage in car capacity	Integer
GarageArea	Size of garage in square feet	Integer
TotalBsmtSF	Total square feet of basement area	Integer
1stFlrSF	First Floor square feet	Binary Integer
FullBath	Full bathrooms above grade	Integer
TotRmsAbvGrd	Total rooms above grade (does not include bathrooms)	Integer
YearBuilt	Original construction date	Integer

After having derived the most correlated features with respect to the target variable SalePrice, it’s imperative to look for how these features behave with respect to SalePrice. In order to achieve the same, we make use of two graphs Joint Plot and Box Plot which are provided by the seaborn library. The visualization provided by these two plots will then form the basis for outlier removal and imputation that is to be performed in the section Data pre-processing. The following graphs have been implemented in order to achieve the same:



 

Figure II: Box Plot between OverallQual and SalePrice

                           
Figure III: Joint Plot between LivingArea and SalePrice

 
Figure IV: Box Plot between GarageCars and SalePrice
 
Figure V: Joint Plot Total Basement Area and SalePrice
 
Figure VI: Joint Plot between First Floor Area and SalePrice


           
Figure VII: Box Plot between Total Rooms Above Grade and SalePrice


 
Figure VIII: Box Plot between Total Rooms and SalePrice

	Data Pre – Processing
Data pre-processing is a technique which is used to convert unprocessed data into meaningful data. There are numerous steps associated with data pre-processing out of which we’re going to be mainly focusing on two of them being Outlier removal and Imputation. Outlier removal is very key to getting high accuracy. Outliers are data points which deviate significantly from the other data points. This may be due to an error in measurements or a one of event. Removal of these outliers makes our dataset more coherent and hence it aids the model to perform better. After analysing the plots obtained in the data visualization section, outliers were identified and thus removed accordingly. Imputation is the process of filling up of missing values present inside a dataset. It is used to increase precision, accuracy and it results in forming a robust statistical model. There are different ways in which imputation is carried out that is Mean, Mode and Median. All the attributes that are of object or string type having null values are imputed with “None” whereas all the attributes that are of int or float type having null values are imputed using the Mode technique. All this data pre-processing helps in improving the generalizability of the model and explicitly expose how the data should be structured based on domain knowledge. Hence, our model is expected to perform much better after the same.

	Model deployment
The proposed model has an implementation of the given machine learning techniques namely being kernel ridge, lasso, elastic net, LightGBM, gradient boosting, and XGBoost regression models. In the text given below, you’ll find the brief description of all these techniques. Further on, an ensemble machine learning model has been proposed which makes use of the given techniques in the following manner:
 
	Lasso model: Lasso regression model stands for Least Absolute Shrinkage and Selection Operator. It’s one of the types of linear regression that makes use of shrinkage technique which then is used to minimize overfitting issues. In the given technique, coefficient of determination is diminished towards zero. It reduces model complexity and prevents over fitting. It’s best suited for multi-collinearity existence in the dataset. Lasso regression executes L1 regularization which then adds a corresponding penalty equal to the absolute value of the magnitude of the coefficients. The main objective of the model is to minimize:
                        
∑_(i=1)^n▒〖(y_(i -  ) ∑_j▒x_ij  β_j)〗^2 + λ∑_(j=1)^p▒〖|β_j |〗
                                         	
	Elastic net model: Linear regression is one of the standard algorithms for regression in which a linear relationship is established between inputs and the target variable. Elastic net is an extension of the linear regression and it combines two popular penalties, the L1 and L2 penalty functions. L1 stands for least absolute deviation also known as LAD. L1 regularization estimates the median of the data. L2 stands for least squared error also known as LS. L2 regularization estimates the mean of the data. 

     L1 Loss Function= ∑_(i=1)^n▒〖|y_true- y_predicted 〗|                                      
                                
L2 Loss Function= ∑_(i=1)^n▒〖(y_true-y_predicted)〗^2 
	Kernel ridge model: Kernel ridge combines the ridge regression model with the kernel trick. Kernel ridge uses L2 norm regularization and employs kernel for space. 

	Gradient Boosting model:  It’s one of the ensemble decision tree models. It is a method which is used to convert weak learners to strong learners. It identifies the shortcomings through loss function. It trains mini models in a gradual, sequential and additive manner. Boosting algorithm makes use of an ensemble model like this one:
                  
                     f(x)= ∑_(m=0)^M▒〖f_m (x)= f_0 (x)+ ∑_(m=1)^M▒〖θ_m φ_m 〗〗(x)                     

	XGBoost model: XGBoost basically improves the GBM framework through system optimization and algorithm enhancement. System optimization comprise of Parallelization, Tree Pruning, Hardware Optimization whereas algorithm enhancements comprise of Regularization, Sparsity awareness and cross-validation.

	LightGBM model: LightGBM regression adds automatic feature selection on boosting examples with large gradient. It identifies the most relevant features of the dataset in reference to the target variable. It increases the performance and training of the machine. As it’s based on decision tree algorithm, it splits the given tree leaf wise according to the best fit whereas other boosting algorithms spilt the tree level wise. Leaf wise approach reduces significantly more losses when compared to the level wise approach. Hence, resulting in better accuracy and also, it’s very fast thus the word “light” finds its mention in the name.

Result
Evaluating a model’s performance is of significant importance as it helps in making the decision as to which model performs better and then selecting that model. It expresses how a model performs in real time using a particular numerical value. There are overall ten model performance techniques out of which chi-square, confusion matrix, cross validation are a few of them. For our analysis, we’re going to be using RMSE (Root mean squared error) and Accuracy. RMSE is a measure of the average squared difference between the estimated values and the actual value. The formula for the same is given as:
                              RMSE= √((∑_(i=1)^N▒〖(〖Predicted〗_i-〗 〖〖Actual〗_i)〗^2)/N  )
Accuracy is a measurement used to determine the best performing model when it comes to identify the relationship between the features of a dataset.
The RMSE values for the adopted techniques and the proposed model on the train data is provided in the table given below:
S.No.	Model used 	RMSE Score on Train Data
1.	Lasso model	0.1111
2.	Elastic Net model	0.1111
3.	Kernel Ridge model	0.1148
4.	Gradient Boosting model	0.1155
5.	XGBoost model	0.1182
6.	LightGBM model	0.1163
7.	Stacked Averaging model (base_model = ENet, GBoost, KRR and meta_model = lasso) 	0.07348652056507986
8.	Proposed Ensemble model	0.0682179145328982

Conclusion
A model with significantly lower RMSE value is developed using ensemble and stacking machine learning algorithms. This model can be put to use in industries as well as by individuals entering the real estate market. Stacking is used with Lasso, Elastic Net, Kernel Ridge GBoost models. Our proposed model is based on an ensemble method comprising of the given stacked model stated above, LightGBM and XGBoost. In comparison to the pre-existing literature, the proposed model outperforms when it comes to RMSE score and accuracy. From our analysis, we concluded that the Lasso model performed the best and hence was selected as the meta model in the stacked technique. The proposed model performs satisfactorily, accurately and up to the mark when it comes to predicting housing prices. With the onset of innovation in the field of computer science, artificial intelligence and machine learning; we expect to see new techniques being deployed to solve the real estate housing problem.

References
[1]	Q. Truong, M. Nguyen, H. Dang, and B. Mei, “Housing Price Prediction via Improved Machine Learning Techniques,” Procedia Computer Science, vol. 174, no. 2019, pp. 433–442, 2020, doi: 10.1016/j.procs.2020.06.111.
[2]	P. F. Pai and W. C. Wang, “Using machine learning models and actual transaction data for predicting real estate prices,” Applied Sciences (Switzerland), vol. 10, no. 17, pp. 1–11, 2020, doi: 10.3390/app10175832.
[3]	M. Modi, A. Sharma, and P. Madhavan, “Applied research on house price prediction using diverse machine learning techniques,” International Journal of Scientific and Technology Research, vol. 9, no. 4, pp. 371–376, 2020.
[4]	H. Yu and J. Wu, “Real Estate Price Prediction with Regression and Classification,” CS 229 Autumn 2016 Project Final Report, pp. 1–5, 2016, [Online]. Available: http://cs229.stanford.edu/proj2016/report/WuYu_HousingPrice_report.pdf
[5]	S. Mullainathan and J. Spiess, “Machine learning: An applied econometric approach,” Journal of Economic Perspectives, vol. 31, no. 2, pp. 87–106, 2017, doi: 10.1257/jep.31.2.87.
[6]	R. Manjula, S. Jain, S. Srivastava, and P. Rajiv Kher, “Real estate value prediction using multivariate regression models,” IOP Conference Series: Materials Science and Engineering, vol. 263, no. 4, 2017, doi: 10.1088/1757-899X/263/4/042098.
[7]	D. Banerjee and S. Dutta, “Predicting the housing price direction using machine learning techniques,” IEEE International Conference on Power, Control, Signals and Instrumentation Engineering, ICPCSI 2017, pp. 2998–3000, 2018, doi: 10.1109/ICPCSI.2017.8392275.
[8]	N. N. Ghosalkar and S. N. Dhage, “Real Estate Value Prediction Using Linear Regression,” Proceedings - 2018 4th International Conference on Computing, Communication Control and Automation, ICCUBEA 2018, pp. 1–5, 2018, doi: 10.1109/ICCUBEA.2018.8697639.
[9]	A. Varma, A. Sarma, S. Doshi, and R. Nair, “House Price Prediction Using Machine Learning and Neural Networks,” Proceedings of the International Conference on Inventive Communication and Computational Technologies, ICICCT 2018, pp. 1936–1939, 2018, doi: 10.1109/ICICCT.2018.8473231.
[10]	T. D. Phan, “Housing price prediction using machine learning algorithms: The case of Melbourne city, Australia,” Proceedings - International Conference on Machine Learning and Data Engineering, iCMLDE 2018, pp. 8–13, 2019, doi: 10.1109/iCMLDE.2018.00017.
[11]	J. I. Pérez-Rave, F. González-Echavarría, and J. C. Correa-Morales, “Modeling of apartment prices in a colombian context from a machine learning approach with stable-important attributes,” DYNA (Colombia), vol. 87, no. 212, pp. 63–72, 2020, doi: 10.15446/dyna.v87n212.80202.
[12]	S. B. Jha, V. Pandey, R. K. Jha, and R. F. Babiceanu, “Machine Learning Approaches to Real Estate Market Prediction Problem: A Case Study,” 2020, [Online]. Available: http://arxiv.org/abs/2008.09922
[13]	O. Al-Gbury and S. Kurnaz, “Real Estate Price Range Prediction Using Artificial Neural Network and Grey Wolf Optimizer,” 4th International Symposium on Multidisciplinary Studies and Innovative Technologies, ISMSIT 2020 - Proceedings, pp. 1–5, 2020, doi: 10.1109/ISMSIT50672.2020.9254972.
[14]	H. Kang and H. Zhao, “Description and Application Research of Multiple Regression Model Optimization Algorithm Based on Data Set Denoising,” Journal of Physics: Conference Series, vol. 1631, no. 1, 2020, doi: 10.1088/1742-6596/1631/1/012063.
[15]	Y. Kang et al., “Understanding house price appreciation using multi-source big geo-data and machine learning,” Land Use Policy, no. July, p. 104919, 2020, doi: 10.1016/j.landusepol.2020.104919.
[16]	G. Uzut and S. Buyrukoglu, “Prediction of real estate prices with data mining algorithms,” Euroasia Journal of Mathematics, Engineering, Natural & Medical Sciences International Indexed & Refereed, vol. 8, no. 9, 2020.
[17]	A. Chauhan, M. Arora, R. Jain, and P. Nagrath, “Evaluation of Real Estate Appraisal Using Ensemble Methods,” SSRN Electronic Journal, pp. 1–5, 2021, doi: 10.2139/ssrn.3747575.
[18]	S. Chaturvedi, L. Ahlawat, T. Patel, and S. Chaturvedi, “EasyChair Preprint № 4926 Real Estate Price Prediction,” 2021.
[19]	E. Tripathi, “Understanding Real Estate Price Prediction using Machine Learning,” International Journal for Research in Applied Science and Engineering Technology, vol. 9, no. 4, pp. 811–816, 2021, doi: 10.22214/ijraset.2021.33720.
[20]	P. Kangane, A. Mallya, A. Gawane, V. Joshi, and S. Gulve, “ANALYSIS OF DIFFERENT REGRESSION MODELS FOR REAL ESTATE PRICE,” vol. 5, no. 11, pp. 247–254, 2021.

