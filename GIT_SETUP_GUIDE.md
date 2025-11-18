# Git Setup Guide

## Step 1: Initial Setup (Already Done!)

✅ Git repository initialized
✅ Branch renamed to 'main'
✅ .gitignore created (protecting .venv, data files, etc.)

## Step 2: Create GitHub Repository

### Option A: Create on GitHub Website (Recommended)

1. Go to https://github.com/new
2. Fill in:
   - Repository name: `airbnb-pricing-nyc` (or your choice)
   - Description: "Machine learning model for NYC Airbnb price prediction (97.2% R² accuracy)"
   - **Keep it Public or Private** (your choice)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
3. Click "Create repository"
4. **Copy the repository URL** (looks like: `https://github.com/YOUR_USERNAME/airbnb-pricing-nyc.git`)

### Option B: Using GitHub CLI (if you have `gh` installed)

```bash
gh repo create airbnb-pricing-nyc --public --source=. --remote=origin
```

## Step 3: Connect Local to GitHub

After creating the GitHub repo, run these commands:

```bash
# Add remote repository (replace URL with yours!)
git remote add origin https://github.com/YOUR_USERNAME/airbnb-pricing-nyc.git

# Verify remote was added
git remote -v
```

## Step 4: Make First Commit

```bash
# Stage all files
git add .

# Check what will be committed (should NOT see .venv, data files, models)
git status

# Commit with message
git commit -m "Initial commit: NYC Airbnb price prediction ML model"

# Push to GitHub
git push -u origin main
```

## What Gets Committed (Safe):

✅ Source code (src/, app/, api/)
✅ Documentation (README.md, MODEL_DOCUMENTATION.md)
✅ Configuration (requirements.txt, .gitignore)
✅ Scripts (setup.sh, start_*.sh)
✅ Tests (tests/)

## What Does NOT Get Committed (Protected by .gitignore):

❌ .venv/ (virtual environment)
❌ .claude/ (IDE cache)
❌ data/raw/*.csv.gz (large data files)
❌ data/processed/*.csv (generated data)
❌ models/*.joblib (large model file)
❌ __pycache__/ (Python cache)

## Future Workflow

After making changes:

```bash
# Check what changed
git status

# Add specific files or all changes
git add filename.py
# OR
git add .

# Commit with descriptive message
git commit -m "Add feature: XYZ"

# Push to GitHub
git push
```

## Common Git Commands

```bash
# See status
git status

# See commit history
git log --oneline

# See what changed
git diff

# Create new branch
git checkout -b feature-name

# Switch branch
git checkout main

# Pull latest changes
git pull

# Undo unstaged changes
git restore filename.py

# Undo last commit (keeps changes)
git reset --soft HEAD~1
```

## Troubleshooting

### "Permission denied" when pushing

You may need to set up authentication:
1. **Personal Access Token** (recommended):
   - Go to GitHub → Settings → Developer settings → Personal access tokens
   - Generate new token with "repo" scope
   - Use token as password when git asks

2. **SSH Keys** (alternative):
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   # Add key to GitHub: Settings → SSH and GPG keys
   ```

### "Repository not found"

Check that:
1. Repository exists on GitHub
2. URL is correct: `git remote -v`
3. You have access to the repository

### Large files warning

If you accidentally try to commit large files:
```bash
# Remove from staging
git reset HEAD large-file.csv

# Make sure it's in .gitignore
echo "large-file.csv" >> .gitignore
```

## Adding Collaborators

1. Go to your GitHub repository
2. Settings → Collaborators
3. Add by username or email

## Best Practices

1. ✅ Commit often with clear messages
2. ✅ Pull before you push
3. ✅ Never commit sensitive data (.env files, API keys)
4. ✅ Never commit large files (>50MB)
5. ✅ Use branches for new features
6. ✅ Write descriptive commit messages

## Example Commit Messages

Good:
- "Add data validation to ingest.py"
- "Fix prediction error handling"
- "Update README with deployment instructions"

Bad:
- "update"
- "fix stuff"
- "changes"

## Your Next Steps:

1. Create GitHub repository (see Step 2 above)
2. Copy your repository URL
3. Run commands from Step 3 and 4
4. Check GitHub - your code should be there!

---

**Need help?** Check Git documentation: https://git-scm.com/doc
