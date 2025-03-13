# Repository Cleanup and Restructuring

This document outlines the cleanup and restructuring process that was performed on the AstroShield repository to improve its organization, reduce its size, and make it more maintainable.

## Cleanup Summary

The repository was restructured to:

1. Reduce the overall repository size from **2.1GB** to **1.1GB** (48% reduction)
2. Implement Git LFS for large files (models, checkpoints)
3. Organize code into logical components
4. Remove unnecessary files and dependencies
5. Create a backup directory for large files that don't need version control

## Git LFS Implementation

Large files are now managed using Git LFS, which stores the actual file content on a remote server while keeping lightweight references in the Git repository. This significantly reduces the repository size and improves clone/pull performance.

Files tracked with Git LFS include:
- Model files (*.onnx, *.pth, *.pt)
- Checkpoint files (checkpoints/**/*.pt)

For more details on working with Git LFS, see [GIT_LFS_GUIDE.md](GIT_LFS_GUIDE.md).

## Backup Directory

The `backup/` directory contains large files that were removed from the main repository to reduce its size. These files include:

- Kafka binaries and archives
- Backup copies of model files

To restore files from the backup directory, use the provided script:

```bash
./restore_from_backup.sh
```

## Excluded Files

The following types of files are excluded from the repository:

1. Virtual environment directories (venv/, env/, .venv/)
2. Node modules and build artifacts (node_modules/, dist/, .next/)
3. Python cache files (__pycache__/, *.pyc, *.pyo)
4. Large binary files that exceed GitHub's limit
5. IDE-specific files (.idea/, .vscode/)
6. Log files and OS-specific files (logs/, .DS_Store, Thumbs.db)

See the `.gitignore` file for the complete list of excluded files.

## Maintenance Guidelines

To maintain the repository's organization and size:

1. Use Git LFS for any new large files (>10MB)
2. Keep virtual environments outside the repository
3. Don't commit build artifacts or compiled code
4. Regularly clean up unnecessary files
5. Follow the established directory structure for new code

## Future Improvements

Potential future improvements include:

1. Further modularization into separate repositories for specific components
2. Implementing a CI/CD pipeline for automated testing and deployment
3. Creating Docker containers for development and deployment environments
4. Improving documentation with more examples and tutorials 