# END-END Text-Summarization Project

## Workflows in each step

1. Update config.yaml
2. Update params.yaml
3. Update entity
4. Update the configuration manager in src config
5. Update the conponents
6. Update the pipeline
7. Update the main.py
8. Update the app.py


# Steps Involved in the Project:

1. **Data Ingestion:** Downloads data from a specified URL if not locally present and extracts the zip file into a directory.
2. **Data Transformation:** Tokenizes input and target texts using Hugging Face Transformers, handling truncation to meet model requirements.
2. **Data Transformation:** Tokenizes input and target texts using Hugging Face Transformers, handling truncation to meet model requirements.
3. **Data Validation:** Ensures all required files are present in the specified directory and logs the validation status.
4. **Model Training:** Uses the Pegasus model for sequence-to-sequence tasks, configures training parameters, trains the model on the dataset, and saves the trained model and tokenizer.
5. **Model Evaluation:** Processes the dataset in smaller batches, generates summaries with the model and tokenizer, computes ROUGE scores, and saves evaluation metrics to a CSV file.

The project follows a modular pipeline approach, ensuring each step from data ingestion to model evaluation is clearly defined and executed systematically, enhancing maintainability and scalability.


## Commands to Setup Project on Local Machine

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dyavadi8769/Text-Summarization.git
   cd Text-Summarization

2.  **Create a virtual environment and activate it:**
    ```bash
    conda create -p env python==3.8 -y
    conda activate env/ 

3.  **Install the Required Dependecies:**
    ```bash
    pip install -r requirements.txt

4. **Run the Web App:**
    ```bash
    python app.py

# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 
	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optional

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = 

    AWS_ECR_LOGIN_URI = 

    ECR_REPOSITORY_NAME = simple-app

yes!! Now you can start predicting 🙂

# Author:

```bash
Author: Sai Kiran Reddy Dyavadi
Role  : Data Scientist
Email : dyavadi324@gmail.com
```
