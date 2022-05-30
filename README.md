# churn_prediction Application built on H2O AutoML and AWS

#### Source Data:
  https://www.kaggle.com/blastchar/telco-customer-churn

#### Machine Learning Model: 
  Stacked Ensemble model trained using H2O AutoML
  
 
 ## Process Flow
 #### 1. Data Sources
 - SQL database 
 - CSV files stored in Amazon S3

#### 2. Data preprocessing
- Library Import
- EDA
- Missing values and outliers treatment
- Categorical Feature Encoding
- Feature Engineering
- Feature Selection

#### 3. Modeling
- Train-test split
- Initializing H2O
- Building AutoML Model
- Saving Model files
- Metric evaluation


#### 4. Deployment
- Build REST Api using Flask
- Make Dockefile and build the Image
- Test Docker locally
- Pushing the Docker Image to Amazon ECR
  * Create a repository in ECR
  * Push the Docker Image
- Creating a Task on AWS ECS to run the API and testing it using the assigned public url
  * Create a Cluster in ECS
  * Create FARGATE task definition
  * Configure tasks
  * Configure Container in EKS
  * Once the tasks are created, test it using public api 
- Logs can be found in CloudWatch 
 
 

  
 
