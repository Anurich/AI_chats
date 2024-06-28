pipeline {
    agent any
	environment {
        CONTAINER_NAME="model-ai-app"
    }
    stages {
        
        stage('Build model-ai Backend') {
            steps {
		sh 'pwd'
        script {
                    try {
                         sh 'sudo docker inspect --format="{{.Id}}" ${CONTAINER_NAME}'
                         sh 'sudo docker stop ${CONTAINER_NAME}'
                         sh 'sudo docker rm ${CONTAINER_NAME}'
                    }
                    catch(all) {
                        sh 'echo "Got Some Error in netsuite connector module"'
                    }
                }
                //dir('') {
                    sh 'sudo docker system prune -a -f'
                    sh 'sudo docker build --no-cache -t model-ai:latest .'
               // }
            }
        }
        stage('Deploy') {
            steps {
                // Build image and start netsuite container.
                
                //dir('nestuite-connector') {
                    
                        sh 'sudo docker-compose up -d'
                        sh 'echo "Deployed netsuite backend module."'
                     
                //}
            }
        }
    }
}

