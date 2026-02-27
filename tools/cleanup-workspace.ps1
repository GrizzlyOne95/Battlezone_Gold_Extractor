param(
    [Parameter(Mandatory = $false)]
    [string]$RepoRoot = ".",

    [Parameter(Mandatory = $false)]
    [string]$ArchiveRoot = "work/archive/root_scratch",

    [Parameter(Mandatory = $false)]
    [switch]$Apply = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RepoPath {
    param([string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return (Resolve-Path -LiteralPath $PathValue).Path
    }
    return (Resolve-Path -LiteralPath (Join-Path -Path (Get-Location) -ChildPath $PathValue)).Path
}

$repo = Resolve-RepoPath -PathValue $RepoRoot
$archiveBase = if ([System.IO.Path]::IsPathRooted($ArchiveRoot)) {
    $ArchiveRoot
} else {
    Join-Path -Path $repo -ChildPath $ArchiveRoot
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$archiveDir = Join-Path -Path $archiveBase -ChildPath $timestamp

$targets = Get-ChildItem -LiteralPath $repo -File |
    Where-Object { $_.Extension -eq ".txt" }

if ($targets.Count -eq 0) {
    Write-Host "No root-level .txt scratch files found."
    exit 0
}

Write-Host ("Found {0} root-level .txt files." -f $targets.Count)
Write-Host ("Archive destination: {0}" -f $archiveDir)

foreach ($f in $targets) {
    $dest = Join-Path -Path $archiveDir -ChildPath $f.Name
    if ($Apply) {
        if (-not (Test-Path -LiteralPath $archiveDir)) {
            New-Item -ItemType Directory -Force -Path $archiveDir | Out-Null
        }
        Move-Item -LiteralPath $f.FullName -Destination $dest -Force
        Write-Host ("MOVED  {0}" -f $f.Name)
    } else {
        Write-Host ("DRYRUN {0} -> {1}" -f $f.Name, $dest)
    }
}

if (-not $Apply) {
    Write-Host "Dry-run complete. Re-run with -Apply to move files."
}
