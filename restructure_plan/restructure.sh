#!/bin/bash

# Create a .gitignore file to exclude large files and directories
cat > .gitignore << 'EOF'
# Virtual environments
env/
venv/
.venv/
**/env/
**/venv/
**/.venv/

# Node modules
**/node_modules/
**/dist/
**/.next/

# Kafka binaries
kafka_2.13-3.9.0/
kafka_2.13-3.9.0.tgz

# Large model files
models/*.onnx
models/*.pth
models/*.pt
checkpoints/

# Python cache
**/__pycache__/
**/*.pyc
**/*.pyo
**/*.pyd
**/.pytest_cache/
**/.coverage
**/.tox/

# Build artifacts
**/build/
**/dist/
**/*.egg-info/

# IDE files
**/.idea/
**/.vscode/
**/*.swp
**/*.swo

# Logs
**/logs/
**/*.log

# OS files
.DS_Store
Thumbs.db
EOF

echo "Created .gitignore file"

# Create a models directory in the restructure plan
mkdir -p restructure_plan/models_info

# Create a file with information about the models
cat > restructure_plan/models_info/README.md << 'EOF'
# Model Files

The model files have been removed from the repository to reduce its size. 
You can download them from the shared storage location or regenerate them using the training scripts.

## Model List
- launch_evaluator.onnx (12MB)
- stimulation_evaluator.onnx (11MB)
- environmental_evaluator.pth (11MB)
- environmental_evaluator.onnx (11MB)
- track_best.pt (9.9MB)
- signature_model.pth (9.0MB)
- signature_model.onnx (9.0MB)
- analyst_evaluator.onnx (1.8MB)
- stability_model.pth (1.4MB)
- stability_model.onnx (1.4MB)
- maneuver_model.pth (1.4MB)
- maneuver_model.onnx (1.4MB)
- rl_maneuver.onnx (532KB)
- game_theory.onnx (276KB)
- compliance_evaluator.onnx (200KB)
- threat_detector.onnx (152KB)
- physical_model.pth (116KB)
- physical_model.onnx (108KB)

## How to Restore Models

1. Download the models from the shared storage:
   ```
   # Example command
   aws s3 cp s3://astroshield-models/models/ ./models/ --recursive
   ```

2. Or regenerate them using the training scripts:
   ```
   python train_models.py
   ```
EOF

echo "Created models info README"

# Create a README for Kafka setup
mkdir -p restructure_plan/kafka_setup

cat > restructure_plan/kafka_setup/README.md << 'EOF'
# Kafka Setup

The Kafka binaries have been removed from the repository to reduce its size.
You can download and set up Kafka using the following instructions:

## Download Kafka

```bash
wget https://downloads.apache.org/kafka/3.9.0/kafka_2.13-3.9.0.tgz
tar -xzf kafka_2.13-3.9.0.tgz
```

## Start Kafka

```bash
# Start ZooKeeper
./kafka_2.13-3.9.0/bin/zookeeper-server-start.sh ./kafka_2.13-3.9.0/config/zookeeper.properties

# Start Kafka
./kafka_2.13-3.9.0/bin/kafka-server-start.sh ./kafka_2.13-3.9.0/config/server.properties
```

## Create Topics

```bash
./kafka_2.13-3.9.0/bin/kafka-topics.sh --create --topic ss2.data.state-vector --bootstrap-server localhost:9092
./kafka_2.13-3.9.0/bin/kafka-topics.sh --create --topic ss5.conjunction.events --bootstrap-server localhost:9092
./kafka_2.13-3.9.0/bin/kafka-topics.sh --create --topic ss4.ccdm.detection --bootstrap-server localhost:9092
./kafka_2.13-3.9.0/bin/kafka-topics.sh --create --topic ss0.sensor.heartbeat --bootstrap-server localhost:9092
```
EOF

echo "Created Kafka setup README"

# Create a README for frontend setup
mkdir -p restructure_plan/frontend_setup

cat > restructure_plan/frontend_setup/README.md << 'EOF'
# Frontend Setup

The node_modules directory has been removed from the repository to reduce its size.
You can set up the frontend using the following instructions:

## Install Dependencies

```bash
cd frontend
npm install
```

## Start Development Server

```bash
npm run dev
```

## Build for Production

```bash
npm run build
```
EOF

echo "Created frontend setup README"

# Create a README for Python environment setup
mkdir -p restructure_plan/python_env_setup

cat > restructure_plan/python_env_setup/README.md << 'EOF'
# Python Environment Setup

The virtual environments have been removed from the repository to reduce its size.
You can set up the Python environment using the following instructions:

## Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```
EOF

echo "Created Python environment setup README"

# Create a main README for the restructured repository
cat > restructure_plan/README.md << 'EOF'
# AstroShield Repository

This repository has been restructured to reduce its size and improve performance.

## Repository Structure

- `astroshield-integration-package/`: UDL integration package
- `backend/`: Backend services
- `frontend/`: Frontend application
- `models/`: Model code (model files are stored separately)
- `services/`: Microservices
- `src/`: Core source code
- `synthetic_data/`: Test data
- `tests/`: Test suite

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/asttroshield.git
   cd asttroshield
   ```

2. **Set Up Python Environment**:
   See `python_env_setup/README.md`

3. **Set Up Frontend**:
   See `frontend_setup/README.md`

4. **Set Up Kafka**:
   See `kafka_setup/README.md`

5. **Download Models**:
   See `models_info/README.md`

## Development Workflow

1. Make your changes
2. Run tests: `pytest tests/`
3. Commit your changes: `git add . && git commit -m "Your message"`
4. Push to the repository: `git push origin main`

## Large Files Storage

Large files are stored separately and not included in the Git repository:
- Model files: Stored in S3 bucket `s3://astroshield-models/`
- Test data: Stored in S3 bucket `s3://astroshield-test-data/`
EOF

echo "Created main README"

# Create a script to clean the repository
cat > restructure_plan/clean_repo.sh << 'EOF'
#!/bin/bash

# Remove large files and directories
echo "Removing Kafka binaries..."
rm -rf kafka_2.13-3.9.0/
rm -f kafka_2.13-3.9.0.tgz

echo "Removing node_modules..."
rm -rf frontend/node_modules/
rm -rf supabase-test/node_modules/
rm -rf frontend/.next/

echo "Removing virtual environments..."
rm -rf env/
rm -rf venv/
rm -rf .venv/

echo "Removing model binary files..."
mkdir -p models_backup
cp models/*.py models_backup/
mv models/*.onnx models_backup/
mv models/*.pth models_backup/
mv models/*.pt models_backup/

echo "Removing Python cache files..."
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
find . -name "*.pyd" -delete

echo "Cleaning Git history (this may take a while)..."
git gc --aggressive --prune=now

echo "Repository cleaned!"
echo "Model files are backed up in models_backup/"
echo "You may want to run 'git add .gitignore' and commit the changes."
EOF

chmod +x restructure_plan/clean_repo.sh

echo "Created clean_repo.sh script"

# Create a script to set up Git LFS
cat > restructure_plan/setup_git_lfs.sh << 'EOF'
#!/bin/bash

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "Git LFS is not installed. Please install it first:"
    echo "  - macOS: brew install git-lfs"
    echo "  - Ubuntu/Debian: apt-get install git-lfs"
    echo "  - Windows: download from https://git-lfs.github.com/"
    exit 1
fi

# Initialize Git LFS
git lfs install

# Track large file types
git lfs track "*.onnx"
git lfs track "*.pth"
git lfs track "*.pt"
git lfs track "*.jar"
git lfs track "*.tgz"
git lfs track "*.gz"
git lfs track "*.zip"
git lfs track "*.tar"
git lfs track "*.bin"
git lfs track "*.h5"
git lfs track "*.hdf5"
git lfs track "*.parquet"
git lfs track "*.npy"
git lfs track "*.npz"

# Add .gitattributes to the repository
git add .gitattributes

echo "Git LFS setup complete!"
echo "Now you can add your large files back to the repository:"
echo "  git add models/*.onnx"
echo "  git commit -m 'Add model files with Git LFS'"
EOF

chmod +x restructure_plan/setup_git_lfs.sh

echo "Created setup_git_lfs.sh script"

# Create a script to split the repository
cat > restructure_plan/split_repo.sh << 'EOF'
#!/bin/bash

# This script helps split the monolithic repository into smaller repositories

# Create directories for the split repositories
mkdir -p ../astroshield-core
mkdir -p ../astroshield-models
mkdir -p ../astroshield-frontend
mkdir -p ../astroshield-integration

# Copy core files
echo "Copying core files..."
cp -r src ../astroshield-core/
cp -r backend ../astroshield-core/
cp -r services ../astroshield-core/
cp -r tests ../astroshield-core/
cp requirements.txt ../astroshield-core/
cp README.md ../astroshield-core/
cp LICENSE ../astroshield-core/

# Copy model files
echo "Copying model files..."
cp -r models ../astroshield-models/
cp -r checkpoints ../astroshield-models/
cp requirements.txt ../astroshield-models/
cp README.md ../astroshield-models/
cp LICENSE ../astroshield-models/

# Copy frontend files
echo "Copying frontend files..."
cp -r frontend ../astroshield-frontend/
cp README.md ../astroshield-frontend/
cp LICENSE ../astroshield-frontend/

# Copy integration files
echo "Copying integration files..."
cp -r astroshield-integration-package ../astroshield-integration/
cp README.md ../astroshield-integration/
cp LICENSE ../astroshield-integration/

echo "Repository split complete!"
echo "You can now initialize Git repositories in each directory:"
echo "  cd ../astroshield-core && git init"
echo "  cd ../astroshield-models && git init"
echo "  cd ../astroshield-frontend && git init"
echo "  cd ../astroshield-integration && git init"
EOF

chmod +x restructure_plan/split_repo.sh

echo "Created split_repo.sh script"

echo "Restructuring plan created in the 'restructure_plan' directory"
echo "Review the plan and execute the scripts as needed" 