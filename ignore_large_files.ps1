$limitMB = 90
$limitBytes = $limitMB * 1024 * 1024
$gitignorePath = ".\.gitignore"

Write-Host "Scanning for files larger than $limitMB MB..." -ForegroundColor Cyan

# Find all files larger than the limit, excluding the .git directory
$largeFiles = Get-ChildItem -Path . -Recurse -File -Force | Where-Object { 
    $_.FullName -notmatch "\\.git\\" -and $_.Length -gt $limitBytes 
}

if ($largeFiles.Count -eq 0) {
    Write-Host "No files larger than $limitMB MB found." -ForegroundColor Green
    exit
}

Write-Host "Found $($largeFiles.Count) large files:" -ForegroundColor Yellow

# Ensure .gitignore exists
if (-not (Test-Path $gitignorePath)) {
    New-Item -Path $gitignorePath -ItemType File | Out-Null
}

# Read existing entries to avoid duplicates
$existingEntries = @()
if (Test-Path $gitignorePath) {
    $existingEntries = Get-Content $gitignorePath
}

$addedCount = 0

foreach ($file in $largeFiles) {
    # Get relative path with forward slashes for gitignore
    $relativePath = $file.FullName.Replace((Get-Location).Path + "\", "").Replace("\", "/")
    
    Write-Host " - $relativePath ($([math]::Round($file.Length / 1MB, 2)) MB)"
    
    # Check if Git already ignores this file (this handles folder wildcards correctly)
    git check-ignore -q $relativePath
    $isIgnored = ($LASTEXITCODE -eq 0)

    if (-not $isIgnored) {
        # Check if file doesn't end with newline, then add one before appending
        $fileContent = Get-Content $gitignorePath -Raw
        if ($fileContent -and -not $fileContent.EndsWith("`n")) {
            $utf8NoBom = New-Object System.Text.UTF8Encoding $False
            [System.IO.File]::AppendAllText((Join-Path (Get-Location) ".gitignore"), "`r`n", $utf8NoBom)
        }

        # Append using UTF-8 without BOM to prevent Git from thinking the entire file changed
        $fullPath = Join-Path (Get-Location) ".gitignore"
        $utf8NoBom = New-Object System.Text.UTF8Encoding $False
        [System.IO.File]::AppendAllLines($fullPath, [string[]]@($relativePath), $utf8NoBom)
        
        $addedCount++
    }
}

if ($addedCount -gt 0) {
    Write-Host "`nSuccessfully added $addedCount new entries to .gitignore" -ForegroundColor Green
} else {
    Write-Host "`nAll large files are already in .gitignore" -ForegroundColor Green
}
