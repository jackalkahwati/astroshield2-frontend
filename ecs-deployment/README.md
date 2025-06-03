# 🚀 AstroShield AWS ECS Deployment

Deploy the complete AstroShield satellite monitoring platform to Amazon Web Services using ECS (Elastic Container Service) with production-ready infrastructure.

## 📋 **Prerequisites**

### **Required Tools**
- [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
- [Docker](https://docs.docker.com/get-docker/)
- bash shell (Linux/macOS/WSL)

### **AWS Requirements**
- AWS Account with administrative permissions
- AWS CLI configured with credentials (`aws configure`)
- Docker running locally

### **Estimated Costs**
- **Development**: ~$30-50/month
- **Production**: ~$100-200/month
- Includes: ECS Fargate, RDS PostgreSQL, ElastiCache Redis, ALB, VPC

## 🎯 **Quick Start (3 Commands)**

```bash
# 1. Clone and navigate
git clone https://github.com/jackalkahwati/astroshield-production.git
cd astroshield-production/ecs-deployment

# 2. Configure deployment
cp configs/deployment.env.template configs/deployment.env
# Edit deployment.env with your settings

# 3. Deploy!
./scripts/deploy-to-ecs.sh
```

**Deployment takes ~15-20 minutes**

## 🏗️ **Architecture Overview**

### **Infrastructure Components**
- **ECS Fargate Cluster**: Serverless container hosting
- **Application Load Balancer**: Traffic distribution and SSL termination
- **RDS PostgreSQL**: Primary database
- **ElastiCache Redis**: Caching layer
- **VPC with NAT Gateway**: Secure networking
- **CloudWatch**: Logging and monitoring

### **Services Deployed**
1. **Frontend** (Next.js) - Port 3000
2. **Backend API** (FastAPI) - Port 3001  
3. **Grafana** (Monitoring) - Port 3000
4. **Prometheus** (Metrics) - Port 9090

### **Network Architecture**
```
Internet → ALB → Private Subnets → ECS Services
                    ↓
            RDS + ElastiCache (Private)
```

## ⚙️ **Configuration**

### **1. Basic Configuration**
```bash
# Copy template
cp configs/deployment.env.template configs/deployment.env

# Edit with your preferred editor
nano configs/deployment.env
```

### **2. Required Settings**
```bash
# AWS Configuration
AWS_REGION=us-west-2
ENVIRONMENT=prod

# UDL Integration (Update with your real endpoints)
UDL_BASE_URL=https://your-udl-service.com/api/v1
UDL_USERNAME=your_username
UDL_PASSWORD=your_password
```

### **3. Optional Security Settings**
```bash
# Auto-generated if left empty (recommended)
DB_PASSWORD=
SECRET_KEY=
GRAFANA_ADMIN_PASSWORD=
```

## 🚀 **Deployment Process**

### **Full Deployment**
```bash
# Load your configuration
source configs/deployment.env

# Deploy complete platform
./scripts/deploy-to-ecs.sh
```

### **What the Script Does**
1. ✅ **Validates** AWS credentials and Docker
2. 📦 **Creates** ECR repositories
3. 🔨 **Builds** and pushes Docker images
4. 🏗️ **Deploys** CloudFormation infrastructure
5. 📋 **Registers** ECS task definitions
6. 🚀 **Creates** ECS services
7. ⏳ **Waits** for services to become stable
8. 🎉 **Outputs** access URLs and credentials

### **Deployment Output**
```
🎉 AstroShield ECS Deployment Complete!

📊 Service URLs:
┌─────────────────────────────────────────────────────────────┐
│  🖥️  Frontend Dashboard:     http://your-alb-url.com       │
│  🔧 Backend API:             http://your-alb-url.com/api   │
│  📚 API Documentation:       http://your-alb-url.com/docs  │
│  📊 Grafana Monitoring:      http://your-alb-url.com/grafana│
└─────────────────────────────────────────────────────────────┘

🔑 Important Credentials:
Database Password: [auto-generated]
Secret Key: [auto-generated]  
Grafana Admin Password: [auto-generated]
```

## 🛠️ **Management Commands**

### **View Services**
```bash
aws ecs list-services --cluster astroshield-prod-cluster --region us-west-2
```

### **Scale Services**
```bash
# Scale frontend to 2 instances
aws ecs update-service \
  --cluster astroshield-prod-cluster \
  --service prod-astroshield-frontend \
  --desired-count 2 \
  --region us-west-2
```

### **View Logs**
```bash
# List log groups
aws logs describe-log-groups \
  --log-group-name-prefix '/ecs/prod-astroshield' \
  --region us-west-2

# Stream logs
aws logs tail /ecs/prod-astroshield-frontend --follow --region us-west-2
```

### **Check Service Health**
```bash
# View service status
aws ecs describe-services \
  --cluster astroshield-prod-cluster \
  --services prod-astroshield-frontend \
  --region us-west-2
```

## 🔒 **Security Features**

### **Network Security**
- ✅ **Private Subnets**: All services run in private subnets
- ✅ **Security Groups**: Least-privilege access rules
- ✅ **NAT Gateway**: Secure outbound internet access
- ✅ **ALB**: Public traffic only through load balancer

### **Data Security**
- ✅ **Encrypted Storage**: RDS and EBS encryption at rest
- ✅ **Secrets Management**: Auto-generated secure passwords
- ✅ **VPC Isolation**: Database not accessible from internet
- ✅ **IAM Roles**: Minimal required permissions

### **Application Security**
- ✅ **Container Scanning**: ECR image vulnerability scanning
- ✅ **Health Checks**: Automatic unhealthy container replacement
- ✅ **Resource Limits**: CPU/memory constraints prevent resource exhaustion

## 📊 **Monitoring & Observability**

### **Built-in Monitoring**
- **Grafana Dashboard**: http://your-alb-url.com/grafana
- **Prometheus Metrics**: Internal service discovery
- **CloudWatch Logs**: Centralized log aggregation
- **ALB Access Logs**: Traffic analysis

### **Health Endpoints**
- **Frontend**: `/`
- **Backend**: `/health`
- **Grafana**: `/api/health`
- **Prometheus**: `/-/healthy`

### **Key Metrics Tracked**
- Service availability and response times
- Container resource utilization
- Database performance
- Cache hit rates
- Load balancer metrics

## 💰 **Cost Optimization**

### **Included Optimizations**
- **Fargate Spot**: 70% cost reduction for non-critical workloads
- **t3.micro instances**: Right-sized for development/small prod
- **Auto-scaling**: Scale down during low usage
- **7-day log retention**: Reduced CloudWatch costs

### **Additional Savings**
```bash
# Use smaller instances for development
export ENVIRONMENT=dev
export DB_INSTANCE_CLASS=db.t3.micro
export CACHE_INSTANCE_TYPE=cache.t3.micro

./scripts/deploy-to-ecs.sh
```

## 🔧 **Troubleshooting**

### **Common Issues**

**1. Docker Build Fails**
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t astroshield-frontend .
```

**2. Service Won't Start**
```bash
# Check service events
aws ecs describe-services \
  --cluster astroshield-prod-cluster \
  --services prod-astroshield-frontend \
  --region us-west-2

# Check container logs
aws logs tail /ecs/prod-astroshield-frontend --region us-west-2
```

**3. Load Balancer Health Check Fails**
```bash
# Check target group health
aws elbv2 describe-target-health \
  --target-group-arn [TARGET_GROUP_ARN] \
  --region us-west-2
```

**4. Database Connection Issues**
```bash
# Verify database is running
aws rds describe-db-instances \
  --db-instance-identifier prod-astroshield-db \
  --region us-west-2

# Check security groups allow port 5432
```

### **Get Support**
- Check AWS CloudFormation events for infrastructure issues
- Review ECS service events for container problems
- Use CloudWatch logs for application debugging
- Verify security group rules for network issues

## 🗑️ **Cleanup**

### **Remove All Resources**
```bash
# WARNING: This deletes everything!
./scripts/cleanup-ecs.sh
```

**What gets deleted:**
- ECS services and cluster
- CloudFormation stack (VPC, ALB, RDS, ElastiCache)
- ECR repositories and images
- CloudWatch log groups
- Local Docker images

## 📁 **File Structure**

```
ecs-deployment/
├── infrastructure/
│   └── astroshield-infrastructure.yml    # CloudFormation template
├── task-definitions/
│   ├── frontend-task.json               # Frontend ECS task
│   ├── backend-task.json                # Backend ECS task
│   ├── grafana-task.json                # Grafana ECS task
│   └── prometheus-task.json             # Prometheus ECS task
├── scripts/
│   ├── deploy-to-ecs.sh                 # Main deployment script
│   └── cleanup-ecs.sh                   # Cleanup script
├── configs/
│   └── deployment.env.template          # Configuration template
└── README.md                            # This file
```

## 🎯 **Next Steps**

### **Post-Deployment**
1. **Set up DNS**: Point your domain to the ALB
2. **Configure SSL**: Add SSL certificate to ALB
3. **Set up monitoring**: Configure Grafana dashboards
4. **Backup strategy**: Set up automated RDS snapshots
5. **CI/CD**: Set up automated deployments

### **Production Hardening**
- Enable AWS Config for compliance monitoring
- Set up AWS GuardDuty for threat detection
- Configure AWS Backup for automated backups
- Implement AWS Secrets Manager for sensitive data
- Set up multi-AZ database deployment

---

## 🆘 **Need Help?**

### **Quick Commands Reference**
```bash
# Deploy
./scripts/deploy-to-ecs.sh

# Monitor
aws ecs list-services --cluster astroshield-prod-cluster

# Scale
aws ecs update-service --cluster astroshield-prod-cluster --service SERVICE_NAME --desired-count 2

# Logs
aws logs tail /ecs/prod-astroshield-frontend --follow

# Cleanup
./scripts/cleanup-ecs.sh
```

**AstroShield ECS deployment provides a production-ready, scalable, and secure satellite monitoring platform on AWS!** 🛰️✨ 