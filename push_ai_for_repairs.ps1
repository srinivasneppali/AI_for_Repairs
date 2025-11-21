# push_ai_for_repairs.ps1
# Usage:
#   Right-click → "Run with PowerShell"
#   or from PowerShell:  .\push_ai_for_repairs.ps1 "Updated WM flow"

param(
    [string]$Message = ""
)

# 1. Go to the repo folder
$repoPath = "C:\Users\srini\Documents\Python Scripts\AI_for_Repairs"
Set-Location $repoPath

# 2. Show current status
git status

# 3. Check if there is anything to commit
$changes = git status --porcelain

if (-not $changes) {
    Write-Host "No changes to commit. Repo is already up to date." -ForegroundColor Yellow
    exit 0
}

# 4. If no message passed, create a timestamp-based one
if ([string]::IsNullOrWhiteSpace($Message)) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $Message = "Auto-commit from script at $timestamp"
}

Write-Host "Using commit message: '$Message'" -ForegroundColor Cyan

# 5. Add everything
git add .

# 6. Commit
git commit -m "$Message"

if ($LASTEXITCODE -ne 0) {
    Write-Host "git commit failed. Aborting push." -ForegroundColor Red
    exit 1
}

# 7. Push to origin/main
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Push completed successfully." -ForegroundColor Green
} else {
    Write-Host "❌ Push failed. Check the error above." -ForegroundColor Red
}
