pipeline {
    agent any

    environment {
        DOCKER_CREDS = credentials('dockerhub-credentials')
        DOCKER_USERNAME = "${DOCKER_CREDS_USR}"
        DOCKER_PASSWORD = "${DOCKER_CREDS_PSW}"
        BACKEND_IMAGE = "${DOCKER_USERNAME}/careershield-backend"
        FRONTEND_IMAGE = "${DOCKER_USERNAME}/careershield-frontend"
        TAG = "v${BUILD_NUMBER}"
    }

    stages {
        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('Retrain & Accuracy Gate') {
            steps {
                script {
                    echo "Running Retraining Pipeline and Accuracy Gate..."
                    sh '''
                    # Ensure venv exists
                    python3 -m venv venv || true
                    . venv/bin/activate
                    pip install -r requirements.txt
                    
                    # Run the pipeline (fails if AUC < 0.70)
                    python scripts/retrain_pipeline.py --data mlops-dataset-layoff-risk/tech_layoffs_2025_2026.csv --models-dir models
                    '''
                }
            }
        }

        stage('Build Docker Images') {
            steps {
                script {
                    echo "Building Backend & Frontend Images..."
                    sh '''
                    docker build -t ${BACKEND_IMAGE}:latest -t ${BACKEND_IMAGE}:${TAG} .
                    docker build -t ${FRONTEND_IMAGE}:latest -t ${FRONTEND_IMAGE}:${TAG} ./frontend
                    '''
                }
            }
        }

        stage('Push to Docker Hub') {
            steps {
                script {
                    echo "Pushing Images to Docker Hub..."
                    sh '''
                    echo "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin
                    docker push ${BACKEND_IMAGE}:latest
                    docker push ${BACKEND_IMAGE}:${TAG}
                    docker push ${FRONTEND_IMAGE}:latest
                    docker push ${FRONTEND_IMAGE}:${TAG}
                    '''
                }
            }
        }

        stage('Deploy via Ansible') {
            steps {
                script {
                    echo "Running Ansible Playbook to update Kubernetes..."
                    sh '''
                    # Passing the new image tags as variables to Ansible
                    ansible-playbook ansible/deploy.yml \
                      -e "backend_image=${BACKEND_IMAGE}:${TAG}" \
                      -e "frontend_image=${FRONTEND_IMAGE}:${TAG}"
                    '''
                }
            }
        }
    }

    post {
        success {
            echo "✅ Pipeline completed successfully! New model deployed to Kubernetes."
        }
        failure {
            echo "❌ Pipeline failed. Check logs for details."
        }
    }
}
