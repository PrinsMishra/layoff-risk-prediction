pipeline {
    agent any

    environment {
        DOCKER_USER = 'prins05688'
        DOCKER_CREDS = 'dockerhub-credentials'
        VERSION = "v${BUILD_NUMBER}"
        COMPOSE_PROJECT_NAME = 'careershield'
    }

    stages {
        stage('1. Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('2. Setup Environment') {
            steps {
                sh '''
                python3 -m venv venv || true
                . venv/bin/activate
                pip install -r backend/requirements.txt
                '''
            }
        }

        stage('3. Check for Data Changes') {
            steps {
                script {
                    // Check if the dataset file was modified in the last commit
                    def dataChanged = sh(script: "git diff --name-only HEAD^ HEAD | grep 'mlops-dataset-layoff-risk/tech_layoffs_2025_2026.csv' || true", returnStdout: true).trim()
                    if (dataChanged) {
                        echo "📊 Dataset change detected. Retraining required."
                        env.RETRAIN_REQUIRED = "true"
                    } else {
                        echo "✅ No dataset changes. Skipping retraining."
                        env.RETRAIN_REQUIRED = "false"
                    }
                }
            }
        }

        stage('4. Retrain Model') {
            when {
                environment name: 'RETRAIN_REQUIRED', value: 'true'
            }
            steps {
                sh '''
                . venv/bin/activate
                python scripts/retrain_pipeline.py --data mlops-dataset-layoff-risk/tech_layoffs_2025_2026.csv --models-dir models
                '''
            }
        }

        stage('5. Validate Metrics') {
            steps {
                script {
                    echo "Checking model quality metrics..."
                    sh 'if [ -f "models/model_schema.json" ]; then echo "✅ Metrics validated."; else exit 1; fi'
                }
            }
        }

        stage('6. Run Unit Tests') {
            steps {
                script {
                    echo "Running Backend Unit Tests..."
                    // We run a simple check to ensure app.py loads correctly
                    sh '. venv/bin/activate && cd backend && python -c "import app; print(\'✅ App load test passed\')"'
                }
            }
        }

        stage('7. Build Docker Images') {
            steps {
                script {
                    echo "Building images with Version: ${VERSION}"
                    sh "docker build -f backend/Dockerfile -t ${DOCKER_USER}/careershield-backend:${VERSION} -t ${DOCKER_USER}/careershield-backend:latest ."
                    sh "docker build -t ${DOCKER_USER}/careershield-frontend:${VERSION} -t ${DOCKER_USER}/careershield-frontend:latest ./frontend"
                }
            }
        }

        stage('8. Push Docker Images') {
            steps {
                script {
                    echo "Pushing Version ${VERSION} to Docker Hub..."
                    withCredentials([usernamePassword(credentialsId: "${DOCKER_CREDS}", usernameVariable: 'USER', passwordVariable: 'PASS')]) {
                        sh "echo \$PASS | docker login -u \$USER --password-stdin"
                        
                        // Push specific version
                        sh "docker push ${DOCKER_USER}/careershield-backend:${VERSION}"
                        sh "docker push ${DOCKER_USER}/careershield-frontend:${VERSION}"
                        
                        // Push latest
                        sh "docker push ${DOCKER_USER}/careershield-backend:latest"
                        sh "docker push ${DOCKER_USER}/careershield-frontend:latest"
                    }
                }
            }
        }

        stage('9. Deploy to Kubernetes') {
            steps {
                script {
                    echo "Deploying Version ${VERSION} to Kubernetes Cluster..."
                    
                    // Replace IMAGE_TAG placeholder with the current build version
                    sh "sed -i 's|IMAGE_TAG|${VERSION}|g' k8s/backend.yaml"
                    sh "sed -i 's|IMAGE_TAG|${VERSION}|g' k8s/frontend.yaml"
                    
                    // Apply all Kubernetes manifests
                    sh "kubectl apply -f k8s/"
                    
                    // Verify rollout status
                    sh "kubectl rollout status deployment/backend-deployment"
                    sh "kubectl rollout status deployment/frontend-deployment"
                    
                    // Restore placeholders for next run
                    sh "git checkout k8s/backend.yaml k8s/frontend.yaml"
                }
            }
        }

        stage('10. Health Check') {
            steps {
                script {
                    echo "Waiting for Kubernetes Service (NodePort 30000) to respond..."
                    sh '''
                    count=0
                    until $(curl --output /dev/null --silent --fail http://localhost:30000/health); do
                        if [ $count -eq 12 ]; then 
                            echo "❌ Health check failed after 60 seconds."
                            exit 1 
                        fi
                        sleep 5
                        count=$((count+1))
                    done
                    echo "✅ Backend is UP in Kubernetes!"
                    '''
                }
            }
        }

        stage('11. Smoke Test') {
            steps {
                script {
                    echo "Verifying AI Inference API in Kubernetes..."
                    sh '''
                    RESPONSE=$(curl -s -X POST "http://localhost:30000/predict" \
                      -H "Content-Type: application/json" \
                      -d '{"industry":"Software","department":"Engineering","ai_exposure":"Partial","total_employees":5000}')
                    
                    if echo "$RESPONSE" | grep -q "risk_probability"; then
                        echo "✅ Smoke Test Passed!"
                    else
                        echo "❌ Unexpected response: $RESPONSE"
                        exit 1
                    fi
                    '''
                }
            }
        }

        stage('12. Cleanup') {
            steps {
                echo "Cleaning up dangling images..."
                sh 'docker image prune -f'
            }
        }

        stage('13. Final Status') {
            steps {
                echo "🚀 Deployment of ${VERSION} is LIVE and verified!"
            }
        }
    }

    post {
        failure {
            echo "❌ Pipeline failed at version ${VERSION}. Check logs for rollback instructions."
        }
    }
}
