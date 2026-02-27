param(
    [Parameter(Mandatory = $true)]
    [string]$GameRoot,

    [Parameter(Mandatory = $false)]
    [string]$OutFile = "notes/file-manifest.csv"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $GameRoot)) {
    throw "GameRoot does not exist: $GameRoot"
}

$root = (Resolve-Path -LiteralPath $GameRoot).Path
$outPath = Join-Path -Path (Get-Location) -ChildPath $OutFile
$outDir = Split-Path -Path $outPath -Parent
if (-not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}

Write-Host "Scanning files under: $root"

$files = Get-ChildItem -LiteralPath $root -Recurse -File
$rows = foreach ($f in $files) {
    $hash = (Get-FileHash -Algorithm SHA1 -LiteralPath $f.FullName).Hash
    $relative = $f.FullName.Substring($root.Length).TrimStart('\\')

    [PSCustomObject]@{
        RelativePath = $relative
        LengthBytes  = $f.Length
        LastWriteUtc = $f.LastWriteTimeUtc.ToString("o")
        SHA1         = $hash
    }
}

$rows | Export-Csv -Path $outPath -NoTypeInformation -Encoding UTF8
Write-Host "Wrote manifest: $outPath"