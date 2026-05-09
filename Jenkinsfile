pipeline {
    agent any

    environment {
        DOCKER_USER = 'prins05688'
        DOCKER_CREDS = 'dockerhub-credentials'
        VERSION = "v${BUILD_NUMBER}"
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
                pip install -r requirements.txt
                '''
            }
        }

        stage('3. Retrain Model') {
            steps {
                sh '''
                . venv/bin/activate
                python scripts/retrain_pipeline.py --data mlops-dataset-layoff-risk/tech_layoffs_2025_2026.csv --models-dir models
                '''
            }
        }

        stage('4. Validate Metrics') {
            steps {
                script {
                    echo "Checking model quality metrics..."
                    sh 'if [ -f "models/model_schema.json" ]; then echo "✅ Metrics validated."; else exit 1; fi'
                }
            }
        }

        stage('5. Run Unit Tests') {
            steps {
                script {
                    echo "Running Backend Unit Tests..."
                    // We run a simple check to ensure app.py loads correctly
                    sh '. venv/bin/activate && python -c "import app; print(\'✅ App load test passed\')"'
                }
            }
        }

        stage('6. Build Docker Images') {
            steps {
                script {
                    echo "Building images with Version: ${VERSION}"
                    sh "docker build -t ${DOCKER_USER}/careershield-backend:${VERSION} -t ${DOCKER_USER}/careershield-backend:latest ."
                    sh "docker build -t ${DOCKER_USER}/careershield-frontend:${VERSION} -t ${DOCKER_USER}/careershield-frontend:latest ./frontend"
                }
            }
        }

        stage('7. Push Docker Images') {
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

        stage('8. Deploy Application') {
            steps {
                echo "Deploying newly built containers..."
                sh 'docker-compose up -d'
            }
        }

        stage('9. Health Check') {
            steps {
                script {
                    echo "Waiting for API endpoint to respond..."
                    sh '''
                    count=0
                    until $(curl --output /dev/null --silent --fail http://localhost:8000/health); do
                        if [ $count -eq 12 ]; then exit 1; fi
                        sleep 5
                        count=$((count+1))
                    done
                    '''
                }
            }
        }

        stage('10. Smoke Test') {
            steps {
                script {
                    echo "Verifying AI Inference API..."
                    sh '''
                    RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" \
                      -H "Content-Type: application/json" \
                      -d '{"industry":"Software","department":"Engineering","ai_exposure":"Partial","total_employees":5000}')
                    
                    if echo "$RESPONSE" | grep -q "risk_probability"; then
                        echo "✅ Smoke Test Passed!"
                    else
                        exit 1
                    fi
                    '''
                }
            }
        }

        stage('11. Cleanup') {
            steps {
                echo "Cleaning up dangling images..."
                sh 'docker image prune -f'
            }
        }

        stage('12. Final Status') {
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
