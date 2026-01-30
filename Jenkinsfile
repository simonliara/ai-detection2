

pipeline {
    agent none
    options {
        timeout(time: 25, unit: 'MINUTES')
        parallelsAlwaysFailFast()
        copyArtifactPermission('**')
    }
    stages {
        stage("Parallel") {
            parallel {
                stage ('Linux') {
                    agent {
                        dockerfile {
                            label 'docker'
                            filename 'ci/linux/Dockerfile'
                            args '-e HOME=/tmp  -v /var/gitcache:/var/gitcache'
                        }
                    }
                    stages {
                        stage ('Run tests') {
                            steps {
                                sh 'nice ci/run_tests.sh'
                            }
                        }
                        stage ('Run Code validation') {
                            steps {
                                sh 'nice ci/run_code_validation.sh'
                            }
                        }
                        stage ('Build whl') {
                            steps {
                                sh 'nice ci/build_package.sh'
                            }
                        }
                    }
                }
            }
         }
    }
}