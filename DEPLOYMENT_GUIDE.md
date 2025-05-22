# AstroShield Secure Deployment Guide

## 🎯 **Current Status: FULLY DEPLOYED & OPERATIONAL**

AstroShield is successfully deployed on AWS EC2 with all services running. Access is provided through secure SSH tunnels following government security best practices.

## 🔐 **Secure Access Instructions**

### **Step 1: Establish SSH Tunnels**
```bash
# Single command to access all AstroShield services
ssh -L 3002:localhost:3001 -L 5002:localhost:5002 -L 3001:localhost:3000 -L 9090:localhost:9090 -N astroshield
```

### **Step 2: Access AstroShield Services**

| Service | Local URL | Description |
|---------|-----------|-------------|
| **🎯 Main Dashboard** | `http://localhost:3002` | **PRIMARY** - Full AstroShield Interface |
| **🔌 API Documentation** | `http://localhost:5002/api/v1/docs` | Interactive API Testing |
| **📊 System Monitoring** | `http://localhost:3001` | Grafana Dashboard |
| **📈 Metrics** | `http://localhost:9090` | Prometheus Metrics |

## 🌟 **Key Features Available**

✅ **Space Surveillance Dashboard**  
✅ **Real-time Satellite Tracking**  
✅ **Trajectory Analysis**  
✅ **Threat Assessment**  
✅ **API Integration**  
✅ **System Monitoring**  
✅ **Database Services (MongoDB, Redis)**  

## 🏗️ **Architecture**

```
[Your Computer] --SSH--> [Bastion] --SSH--> [AstroShield EC2]
     ↓                                           ↓
[localhost:3002] <-- Encrypted Tunnel --> [Next.js:3001]
[localhost:5002] <-- Encrypted Tunnel --> [API:5002]
[localhost:3001] <-- Encrypted Tunnel --> [Grafana:3000]
```

## 🔧 **Deployment Details**

- **Frontend**: Next.js React application
- **Backend**: FastAPI with Python
- **Database**: MongoDB + Redis
- **Monitoring**: Grafana + Prometheus
- **Security**: SSH tunnel encryption
- **Environment**: AWS EC2 with restricted access

## 📋 **For External Teams/Users**

To access AstroShield:
1. Obtain SSH access credentials from system administrator
2. Run the SSH tunnel command above
3. Open `http://localhost:3002` in your browser
4. Begin using the AstroShield platform

**Note**: This secure access pattern is intentional and follows government security protocols for sensitive space surveillance systems. 