# Build a Smart House Price Prediction system using ML, deployed via FastAPI, containerized with Docker, and hosted on AWS App Runner with CI/CD automation.


# Smart House Price Prediction (MLOps Project)

A complete end-to-end **MLOps project** demonstrating how to build, containerize, and deploy a machine learning model using **FastAPI**, **Docker**, and **AWS App Runner**. This project predicts house prices in different locations of bangalore based on features like BHK, square footage, location, and amenities.



## Project Overview
This project showcases:
- Machine Learning model (RandomForestRegressor)
- FastAPI backend with REST API + UI
- Docker containerization
- Deployment on AWS (ECR + App Runner)
- CI/CD automation with GitHub Actions
- Monitoring via CloudWatch



## Architecture Flow
User → FastAPI → ML Model → Docker → ECR → App Runner → CI/CD → CloudWatch



## Getting Started

### 

1. Clone Repository

git clone https://github.com/disisdsid/mlops-project-1
cd mlops-project-1

2. Train Model

python train_house.py
This generates house_model_v1.pkl used by the FastAPI app.


3. Run Locally

uvicorn main_house:app --reload
Visit:

http://127.0.0.1:8000/ → Health check

http://127.0.0.1:8000/ui → Web UI

http://127.0.0.1:8000/predict → API endpoint

http://127.0.0.1:8000/docs → Swagger UI


4. Docker Build

docker build -t house-price-api .

5. Push to AWS ECR

aws ecr create-repository --repository-name house-price-api
docker tag house-price-api:latest <ECR_URI>
docker push <ECR_URI>

6. Deploy on AWS App Runner
Select ECR image

Set port to 8000

Configure health check (recommended: HTTP /health)


 Key API Endpoints
/ → Health check

/ui → User Interface form

/predict → JSON API prediction

/docs → Swagger UI


 CI/CD Pipeline
GitHub Actions automates:

Code push

Docker build

Push to ECR

Deploy to App Runner


 Challenges Faced
Docker daemon issues

Setting up UI with python code

UI route confusion (/ui)



 Improvements & Future Work
Add database integration (DynamoDB)

Use larger real-world dataset

Model versioning with S3

Setup custom domain



 Conclusion
This project demonstrates a complete MLOps lifecycle — from model training to cloud deployment — and is suitable for beginners to understand real-world ML deployment pipelines.


 License
This project is licensed under the MIT License.

Created by `Siddhanta Dash`

LinkedIn - https://www.linkedin.com/in/siddhanta-dash


