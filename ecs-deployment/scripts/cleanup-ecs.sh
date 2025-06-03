#!/bin/bash

# AstroShield ECS Cleanup Script
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

echo -e "${RED}🗑️  AstroShield ECS Cleanup Script${NC}"
echo -e "${YELLOW}⚠️  This will PERMANENTLY DELETE all AstroShield resources!${NC}"
echo -e "${YELLOW}Environment: ${ENVIRONMENT}${NC}"
echo -e "${YELLOW}Region: ${AWS_REGION}${NC}"
echo

# Confirmation
read -p "Are you sure you want to delete all AstroShield resources? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo -e "${GREEN}Cancelled. No resources were deleted.${NC}"
    exit 0
fi

echo -e "${RED}🚨 Final confirmation: Type 'DELETE' to proceed:${NC}"
read -p "> " -r
if [[ $REPLY != "DELETE" ]]; then
    echo -e "${GREEN}Cancelled. No resources were deleted.${NC}"
    exit 0
fi

echo -e "${BLUE}Starting cleanup process...${NC}"

# Check AWS CLI and credentials
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI not found${NC}"
    exit 1
fi

if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS credentials not configured${NC}"
    exit 1
fi

# Get AWS Account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo -e "${GREEN}✅ AWS Account ID: ${AWS_ACCOUNT_ID}${NC}"

# Check if stack exists
if ! aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$AWS_REGION" &> /dev/null; then
    echo -e "${YELLOW}⚠️  CloudFormation stack '$STACK_NAME' not found${NC}"
else
    # Get ECS cluster name
    ECS_CLUSTER=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$AWS_REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`ECSCluster`].OutputValue' \
        --output text 2>/dev/null || echo "")

    if [[ -n "$ECS_CLUSTER" ]]; then
        echo -e "${BLUE}🛑 Stopping ECS services...${NC}"
        
        # List and stop all services in the cluster
        services=$(aws ecs list-services \
            --cluster "$ECS_CLUSTER" \
            --region "$AWS_REGION" \
            --query 'serviceArns' \
            --output text 2>/dev/null || echo "")

        if [[ -n "$services" ]]; then
            for service_arn in $services; do
                service_name=$(basename "$service_arn")
                echo -e "${YELLOW}Stopping service: $service_name${NC}"
                
                # Scale down to 0
                aws ecs update-service \
                    --cluster "$ECS_CLUSTER" \
                    --service "$service_name" \
                    --desired-count 0 \
                    --region "$AWS_REGION" &> /dev/null || true
                
                # Delete service
                aws ecs delete-service \
                    --cluster "$ECS_CLUSTER" \
                    --service "$service_name" \
                    --region "$AWS_REGION" &> /dev/null || true
            done
            
            echo -e "${BLUE}⏳ Waiting for services to stop...${NC}"
            sleep 10
        fi
    fi
    
    # Delete CloudFormation stack
    echo -e "${BLUE}🗑️  Deleting CloudFormation stack...${NC}"
    aws cloudformation delete-stack \
        --stack-name "$STACK_NAME" \
        --region "$AWS_REGION"
    
    echo -e "${BLUE}⏳ Waiting for stack deletion...${NC}"
    aws cloudformation wait stack-delete-complete \
        --stack-name "$STACK_NAME" \
        --region "$AWS_REGION"
    
    echo -e "${GREEN}✅ CloudFormation stack deleted${NC}"
fi

# Delete ECR repositories
echo -e "${BLUE}📦 Deleting ECR repositories...${NC}"

repositories=("astroshield-frontend" "astroshield-backend")

for repo in "${repositories[@]}"; do
    if aws ecr describe-repositories --repository-names "$repo" --region "$AWS_REGION" &> /dev/null; then
        echo -e "${YELLOW}Deleting ECR repository: $repo${NC}"
        aws ecr delete-repository \
            --repository-name "$repo" \
            --region "$AWS_REGION" \
            --force || echo "Failed to delete $repo repository"
    else
        echo -e "${YELLOW}⚠️  ECR repository '$repo' not found${NC}"
    fi
done

# Delete CloudWatch Log Groups
echo -e "${BLUE}📋 Deleting CloudWatch Log Groups...${NC}"

log_groups=$(aws logs describe-log-groups \
    --log-group-name-prefix "/ecs/${ENVIRONMENT}-astroshield" \
    --region "$AWS_REGION" \
    --query 'logGroups[].logGroupName' \
    --output text 2>/dev/null || echo "")

if [[ -n "$log_groups" ]]; then
    for log_group in $log_groups; do
        echo -e "${YELLOW}Deleting log group: $log_group${NC}"
        aws logs delete-log-group \
            --log-group-name "$log_group" \
            --region "$AWS_REGION" || echo "Failed to delete $log_group"
    done
else
    echo -e "${YELLOW}⚠️  No log groups found${NC}"
fi

# Cleanup local Docker images
echo -e "${BLUE}🐳 Cleaning up local Docker images...${NC}"
docker images | grep astroshield | awk '{print $3}' | xargs -r docker rmi -f || true
docker images | grep "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com" | awk '{print $3}' | xargs -r docker rmi -f || true

# Cleanup temporary files
echo -e "${BLUE}🧹 Cleaning up temporary files...${NC}"
rm -f /tmp/*-task-resolved.json

echo
echo -e "${GREEN}🎉 AstroShield ECS Cleanup Complete!${NC}"
echo
echo -e "${BLUE}📊 Cleanup Summary:${NC}"
echo "✅ CloudFormation stack deleted"
echo "✅ ECS services stopped and deleted"
echo "✅ ECR repositories deleted"
echo "✅ CloudWatch log groups deleted"
echo "✅ Local Docker images cleaned"
echo "✅ Temporary files cleaned"
echo
echo -e "${GREEN}All AstroShield resources have been successfully removed.${NC}"
echo -e "${BLUE}💡 You can redeploy anytime using: ./deploy-to-ecs.sh${NC}" 