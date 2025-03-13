# Repository Cleanup and Restructuring

This document provides instructions for cleaning up and restructuring the AstroShield repository to reduce its size and improve performance.

## Quick Fix (Immediate Size Reduction)

For an immediate size reduction, run the `quick_fix.sh` script:

```bash
./quick_fix.sh
```

This script will:
1. Move Kafka binaries to a backup directory
2. Remove node_modules directories
3. Remove virtual environments
4. Backup model binary files
5. Remove Python cache files
6. Create a .gitignore file to prevent these files from being added back
7. Create a README in the backup directory with instructions for restoring files

After running this script, your repository size should be significantly reduced, and your computer should perform better when working with the codebase.

## Comprehensive Restructuring

For a more comprehensive restructuring, explore the options in the `restructure_plan` directory:

### 1. Clean Repository

To clean the repository more thoroughly, including removing large files from Git history:

```bash
./restructure_plan/clean_repo.sh
```

### 2. Set Up Git LFS

To set up Git Large File Storage for handling large files:

```bash
./restructure_plan/setup_git_lfs.sh
```

### 3. Split Repository

To split the monolithic repository into smaller, focused repositories:

```bash
./restructure_plan/split_repo.sh
```

## Best Practices for Repository Management

1. **Keep binary files out of Git**: Use Git LFS or external storage for large binary files.
2. **Use .gitignore**: Ensure that build artifacts, virtual environments, and node_modules are excluded.
3. **Split repositories**: Consider maintaining separate repositories for frontend, backend, models, etc.
4. **Document dependencies**: Ensure that all dependencies are properly documented in requirements.txt or package.json.
5. **Use shallow clones**: When cloning the repository, use `git clone --depth=1` to get only the latest version.

## Restoring Files

If you need to restore files that were removed during cleanup:

1. Check the `backup` directory for files moved there.
2. For Kafka, download from: https://downloads.apache.org/kafka/3.9.0/kafka_2.13-3.9.0.tgz
3. For node_modules, run `npm install` in the respective directories.
4. For virtual environments, create new ones and install dependencies from requirements.txt.

## Questions or Issues

If you encounter any issues with the cleanup or restructuring process, please contact the repository maintainer. 