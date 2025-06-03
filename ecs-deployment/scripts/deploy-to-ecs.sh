#!/bin/bash

# AstroShield ECS Deployment Script
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${ENVIRONMENT:-prod}
AWS_REGION=${AWS_REGION:-us-west-2}
STACK_NAME="astroshield-${ENVIRONMENT}"
DB_PASSWORD=${DB_PASSWORD:-$(openssl rand -base64 32)}
SECRET_KEY=${SECRET_KEY:-$(openssl rand -base64 32)}
GRAFANA_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-$(openssl rand -base64 16)}

# UDL Configuration (set these environment variables)
UDL_BASE_URL=${UDL_BASE_URL:-"https://mock-udl-service.local/api/v1"}
UDL_USERNAME=${UDL_USERNAME:-"mockuser"}
UDL_PASSWORD=${UDL_PASSWORD:-"mockpass"}

echo -e "${BLUE}🚀 Starting AstroShield ECS Deployment...${NC}"
echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"
echo -e "${BLUE}Region: ${AWS_REGION}${NC}"

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI not found. Please install AWS CLI${NC}"
    exit 1
fi

# Check if AWS credentials are configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS credentials not configured. Run 'aws configure'${NC}"
    exit 1
fi

# Get AWS Account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo -e "${GREEN}✅ AWS Account ID: ${AWS_ACCOUNT_ID}${NC}"

# Step 1: Create ECR repositories
echo -e "${BLUE}📦 Creating ECR repositories...${NC}"

repositories=("astroshield-frontend" "astroshield-backend")

for repo in "${repositories[@]}"; do
    if aws ecr describe-repositories --repository-names "$repo" --region "$AWS_REGION" &> /dev/null; then
        echo -e "${YELLOW}⚠️  ECR repository $repo already exists${NC}"
    else
        echo -e "${GREEN}Creating ECR repository: $repo${NC}"
        aws ecr create-repository \
            --repository-name "$repo" \
            --region "$AWS_REGION" \
            --image-scanning-configuration scanOnPush=true
    fi
done

# Step 2: Build and push Docker images
echo -e "${BLUE}🔨 Building and pushing Docker images...${NC}"

# Login to ECR
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

# Build and push frontend
echo -e "${GREEN}Building frontend image...${NC}"
cd ../astroshield-production/frontend
docker build -t "astroshield-frontend:latest" \
    --build-arg NODE_ENV=production \
    --build-arg NEXT_PUBLIC_API_URL="PLACEHOLDER_BACKEND_URL/api/v1" .

docker tag "astroshield-frontend:latest" "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/astroshield-frontend:latest"
docker push "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/astroshield-frontend:latest"

# Build and push backend
echo -e "${GREEN}Building backend image...${NC}"
cd ../../backend_fixed
docker build -t "astroshield-backend:latest" .
docker tag "astroshield-backend:latest" "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/astroshield-backend:latest"
docker push "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/astroshield-backend:latest"

# Return to deployment directory
cd ../ecs-deployment/scripts

# Step 3: Deploy infrastructure
echo -e "${BLUE}🏗️  Deploying infrastructure...${NC}"

aws cloudformation deploy \
    --template-file ../infrastructure/astroshield-infrastructure.yml \
    --stack-name "$STACK_NAME" \
    --parameter-overrides \
        EnvironmentName="$ENVIRONMENT" \
        DBPassword="$DB_PASSWORD" \
    --capabilities CAPABILITY_IAM \
    --region "$AWS_REGION"

# Get stack outputs
echo -e "${GREEN}Getting stack outputs...${NC}"
LOAD_BALANCER_URL=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$AWS_REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerURL`].OutputValue' \
    --output text)

ECS_CLUSTER=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$AWS_REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`ECSCluster`].OutputValue' \
    --output text)

PRIVATE_SUBNETS=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$AWS_REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`PrivateSubnets`].OutputValue' \
    --output text)

ECS_SECURITY_GROUP=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$AWS_REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`ECSSecurityGroup`].OutputValue' \
    --output text)

ECS_EXECUTION_ROLE=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$AWS_REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`ECSTaskExecutionRole`].OutputValue' \
    --output text)

ECS_TASK_ROLE=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$AWS_REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`ECSTaskRole`].OutputValue' \
    --output text)

DB_ENDPOINT=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$AWS_REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`DatabaseEndpoint`].OutputValue' \
    --output text)

REDIS_ENDPOINT=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$AWS_REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`RedisEndpoint`].OutputValue' \
    --output text)

FRONTEND_TARGET_GROUP=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$AWS_REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`FrontendTargetGroup`].OutputValue' \
    --output text)

BACKEND_TARGET_GROUP=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$AWS_REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`BackendTargetGroup`].OutputValue' \
    --output text)

GRAFANA_TARGET_GROUP=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$AWS_REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`GrafanaTargetGroup`].OutputValue' \
    --output text)

echo -e "${GREEN}✅ Infrastructure deployed successfully!${NC}"
echo -e "${GREEN}Load Balancer URL: $LOAD_BALANCER_URL${NC}"

# Step 4: Create and register task definitions
echo -e "${BLUE}📋 Creating ECS task definitions...${NC}"

# Function to replace placeholders in task definition
replace_placeholders() {
    local file=$1
    local output=$2
    
    sed "s|\${ECR_FRONTEND_IMAGE}|$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/astroshield-frontend:latest|g; \
         s|\${ECR_BACKEND_IMAGE}|$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/astroshield-backend:latest|g; \
         s|\${ECS_EXECUTION_ROLE_ARN}|$ECS_EXECUTION_ROLE|g; \
         s|\${ECS_TASK_ROLE_ARN}|$ECS_TASK_ROLE|g; \
         s|\${AWS_REGION}|$AWS_REGION|g; \
         s|\${DB_ENDPOINT}|$DB_ENDPOINT|g; \
         s|\${REDIS_ENDPOINT}|$REDIS_ENDPOINT|g; \
         s|\${DB_USERNAME}|astroshield|g; \
         s|\${DB_PASSWORD}|$DB_PASSWORD|g; \
         s|\${SECRET_KEY}|$SECRET_KEY|g; \
         s|\${FRONTEND_URL}|$LOAD_BALANCER_URL|g; \
         s|\${BACKEND_URL}|$LOAD_BALANCER_URL|g; \
         s|\${FRONTEND_DOMAIN}|$(echo $LOAD_BALANCER_URL | sed 's|http://||')|g; \
         s|\${GRAFANA_ADMIN_PASSWORD}|$GRAFANA_ADMIN_PASSWORD|g; \
         s|\${UDL_BASE_URL}|$UDL_BASE_URL|g; \
         s|\${UDL_USERNAME}|$UDL_USERNAME|g; \
         s|\${UDL_PASSWORD}|$UDL_PASSWORD|g" "$file" > "$output"
}

# Register task definitions
for task in frontend backend grafana prometheus; do
    echo -e "${GREEN}Registering $task task definition...${NC}"
    replace_placeholders "../task-definitions/${task}-task.json" "/tmp/${task}-task-resolved.json"
    
    aws ecs register-task-definition \
        --cli-input-json "file:///tmp/${task}-task-resolved.json" \
        --region "$AWS_REGION"
done

# Step 5: Create ECS services
echo -e "${BLUE}🚀 Creating ECS services...${NC}"

# Create frontend service
aws ecs create-service \
    --cluster "$ECS_CLUSTER" \
    --service-name "${ENVIRONMENT}-astroshield-frontend" \
    --task-definition "astroshield-frontend" \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$PRIVATE_SUBNETS],securityGroups=[$ECS_SECURITY_GROUP],assignPublicIp=DISABLED}" \
    --load-balancers "targetGroupArn=$FRONTEND_TARGET_GROUP,containerName=frontend,containerPort=3000" \
    --region "$AWS_REGION" || echo "Frontend service may already exist"

# Create backend service
aws ecs create-service \
    --cluster "$ECS_CLUSTER" \
    --service-name "${ENVIRONMENT}-astroshield-backend" \
    --task-definition "astroshield-backend" \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$PRIVATE_SUBNETS],securityGroups=[$ECS_SECURITY_GROUP],assignPublicIp=DISABLED}" \
    --load-balancers "targetGroupArn=$BACKEND_TARGET_GROUP,containerName=backend,containerPort=3001" \
    --region "$AWS_REGION" || echo "Backend service may already exist"

# Create grafana service
aws ecs create-service \
    --cluster "$ECS_CLUSTER" \
    --service-name "${ENVIRONMENT}-astroshield-grafana" \
    --task-definition "astroshield-grafana" \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$PRIVATE_SUBNETS],securityGroups=[$ECS_SECURITY_GROUP],assignPublicIp=DISABLED}" \
    --load-balancers "targetGroupArn=$GRAFANA_TARGET_GROUP,containerName=grafana,containerPort=3000" \
    --region "$AWS_REGION" || echo "Grafana service may already exist"

# Create prometheus service (no load balancer, internal only)
aws ecs create-service \
    --cluster "$ECS_CLUSTER" \
    --service-name "${ENVIRONMENT}-astroshield-prometheus" \
    --task-definition "astroshield-prometheus" \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[$PRIVATE_SUBNETS],securityGroups=[$ECS_SECURITY_GROUP],assignPublicIp=DISABLED}" \
    --region "$AWS_REGION" || echo "Prometheus service may already exist"

# Step 6: Wait for services to be stable
echo -e "${BLUE}⏳ Waiting for services to become stable...${NC}"

services=("${ENVIRONMENT}-astroshield-frontend" "${ENVIRONMENT}-astroshield-backend" "${ENVIRONMENT}-astroshield-grafana" "${ENVIRONMENT}-astroshield-prometheus")

for service in "${services[@]}"; do
    echo -e "${YELLOW}Waiting for $service to be stable...${NC}"
    aws ecs wait services-stable \
        --cluster "$ECS_CLUSTER" \
        --services "$service" \
        --region "$AWS_REGION"
    echo -e "${GREEN}✅ $service is stable${NC}"
done

# Step 7: Output final information
echo
echo -e "${GREEN}🎉 AstroShield ECS Deployment Complete!${NC}"
echo
echo -e "${BLUE}📊 Service URLs:${NC}"
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│  🖥️  Frontend Dashboard:     $LOAD_BALANCER_URL           │"
echo "│  🔧 Backend API:             $LOAD_BALANCER_URL/api       │"
echo "│  📚 API Documentation:       $LOAD_BALANCER_URL/docs      │"
echo "│  📊 Grafana Monitoring:      $LOAD_BALANCER_URL/grafana   │"
echo "│       └─ Login: admin/$GRAFANA_ADMIN_PASSWORD             │"
echo "└─────────────────────────────────────────────────────────────┘"
echo
echo -e "${BLUE}🔑 Important Credentials:${NC}"
echo "Database Password: $DB_PASSWORD"
echo "Secret Key: $SECRET_KEY"
echo "Grafana Admin Password: $GRAFANA_ADMIN_PASSWORD"
echo
echo -e "${YELLOW}💾 Save these credentials securely!${NC}"
echo
echo -e "${BLUE}🛠️  Management Commands:${NC}"
echo "View services: aws ecs list-services --cluster $ECS_CLUSTER --region $AWS_REGION"
echo "View logs: aws logs describe-log-groups --log-group-name-prefix '/ecs/${ENVIRONMENT}-astroshield' --region $AWS_REGION"
echo "Scale service: aws ecs update-service --cluster $ECS_CLUSTER --service SERVICE_NAME --desired-count 2 --region $AWS_REGION"
echo
echo -e "${GREEN}✨ AstroShield is now running on AWS ECS!${NC}" 