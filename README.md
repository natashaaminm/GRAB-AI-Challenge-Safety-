# GRAB-AI-Challenge-Safety-
GRAB AI Challenge (Safety) Submission
In this project, I aim to predict if a ride is dangerous or not. Based on my research, dangerous driving vsn be split into 3 key parts. 1) Speeds above the speed limit 2) Excessive acceleration and deceleration 3) Turning at high speeds


New features taking into account these 3 key points were created: 

  Speed above the speed limit in Singapore i.e. greater than 90 km/h ~ 25m/s 
	
 a) Var, Max, Mean of the Moving Average of the Resultant Acceleration:  Resultant Acceleration=(sqrt((accleration_x)^2) + (accleration_y)^2) + (accleration_z)^2))
	
 b) Turning at high speeds is reflected by a combination of 1) High Acceleration and 2) High Change in bearing. To depict this change I use a combination of 
	
	
 c) Due to the data being imbalanced I then use SMOTE to make the data less imbalanced. 
	
	
  Following that, I run 3 key machine learning models with hyperparameter tuning using ROC-AUC as a metric. These 3 models are Random Forest, Logistic Regression and XGBoost. I use these models as they are suited the classification task. 
 
