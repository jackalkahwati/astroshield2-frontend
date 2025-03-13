# Git LFS Guide for AstroShield Repository

## Overview

This repository uses Git Large File Storage (LFS) to efficiently manage large files. Git LFS replaces large files with text pointers inside Git, while storing the file contents on a remote server.

## File Types Tracked by Git LFS

The following file types are currently tracked by Git LFS:

- Model files: `*.onnx`, `*.pth`, `*.pt`
- Archive files: `*.zip`, `*.tar`, `*.tgz`, `*.gz`
- Data files: `*.bin`, `*.h5`, `*.hdf5`, `*.parquet`, `*.npy`, `*.npz`
- Java archives: `*.jar`

## Setup for New Team Members

If you're cloning this repository for the first time, follow these steps:

1. **Install Git LFS**:
   - macOS: `brew install git-lfs`
   - Ubuntu/Debian: `apt-get install git-lfs`
   - Windows: Download from https://git-lfs.github.com/

2. **Clone the repository**:
   ```bash
   git clone https://github.com/jackalkahwati/asttroshield_v0.git
   cd asttroshield_v0
   ```

3. **Initialize Git LFS**:
   ```bash
   git lfs install
   ```

4. **Pull LFS files**:
   ```bash
   git lfs pull
   ```

## Working with Git LFS Files

### Adding New Large Files

When adding new files of the tracked types (e.g., `.onnx`, `.pth`), Git LFS will automatically handle them:

```bash
# Example: Adding a new model file
cp /path/to/new_model.onnx models/
git add models/new_model.onnx
git commit -m "Add new model file"
```

### Adding New File Types to LFS

If you need to track additional file types with Git LFS:

```bash
git lfs track "*.new_extension"
git add .gitattributes
git commit -m "Track *.new_extension with Git LFS"
```

### Checking LFS-tracked Files

To see which files are being tracked by Git LFS:

```bash
git lfs ls-files
```

### Pulling LFS Files

If you need to explicitly pull LFS files:

```bash
git lfs pull
```

## Troubleshooting

### Missing LFS Files

If you're seeing text pointers instead of actual file contents:

```bash
git lfs pull
```

### LFS Files Not Being Tracked

If your large files aren't being tracked by LFS:

1. Check if the file type is in `.gitattributes`:
   ```bash
   cat .gitattributes
   ```

2. If not, add it:
   ```bash
   git lfs track "*.your_extension"
   git add .gitattributes
   git commit -m "Track *.your_extension with Git LFS"
   ```

3. Re-add your file:
   ```bash
   git add your_file.your_extension
   ```

## Best Practices

1. **Always pull LFS files after cloning**: Use `git lfs pull` after cloning the repository.
2. **Keep .gitattributes up to date**: Always commit changes to `.gitattributes` when adding new file types.
3. **Be mindful of bandwidth**: Git LFS files count against your GitHub bandwidth quota.
4. **Consider shallow clones for CI/CD**: Use `git clone --depth=1` for CI/CD pipelines to save bandwidth.

## Additional Resources

- [Git LFS Documentation](https://git-lfs.github.com/)
- [GitHub's Git LFS Guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage) 