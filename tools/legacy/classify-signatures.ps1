param(
    [Parameter(Mandatory = $false)]
    [string]$InputRoot = "work/extracted/full",

    [Parameter(Mandatory = $false)]
    [string]$OutDir = "notes/signatures",

    [Parameter(Mandatory = $false)]
    [int]$HeaderBytes = 16
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-WorkspacePath {
    param([string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return (Resolve-Path -LiteralPath $PathValue).Path
    }
    $candidate = Join-Path -Path (Get-Location) -ChildPath $PathValue
    return (Resolve-Path -LiteralPath $candidate).Path
}

function Get-LikelyType {
    param([byte[]]$Bytes)
    if ($Bytes.Length -lt 4) { return "Unknown" }

    $hex4 = (($Bytes[0..3] | ForEach-Object { "{0:X2}" -f $_ }) -join " ")
    switch ($hex4) {
        "52 49 46 46" { return "RIFF (likely WAV)" }
        "44 44 53 20" { return "DDS texture" }
        "89 50 4E 47" { return "PNG image" }
        default {}
    }

    if ($Bytes.Length -ge 2) {
        $hex2 = (($Bytes[0..1] | ForEach-Object { "{0:X2}" -f $_ }) -join " ")
        if ($hex2 -eq "42 4D") { return "BMP image" }
    }

    if ($Bytes.Length -ge 3) {
        $hex3 = (($Bytes[0..2] | ForEach-Object { "{0:X2}" -f $_ }) -join " ")
        if ($hex3 -eq "FF D8 FF") { return "JPEG image" }
    }

    if ($Bytes.Length -ge 8) {
        $ascii8 = [System.Text.Encoding]::ASCII.GetString($Bytes, 0, 8)
        if ($ascii8 -eq "AsuraCmp") { return "Asura compressed container" }
        if ($ascii8 -eq "AsuraZlb") { return "Asura zlib container" }
        if ($ascii8 -eq "AsuraZbb") { return "Asura chunked zlib container" }
        if ($ascii8 -eq "Asura   ") { return "Asura placeholder/stub" }
    }

    if ($Bytes.Length -ge 4) {
        $tga = (($Bytes[0..3] | ForEach-Object { "{0:X2}" -f $_ }) -join " ")
        if ($tga -eq "00 00 02 00") { return "TGA image (uncompressed true-color)" }
    }

    return "Binary/unknown"
}

$inputRoot = Resolve-WorkspacePath -PathValue $InputRoot
$outRootAbs = if ([System.IO.Path]::IsPathRooted($OutDir)) {
    $OutDir
} else {
    Join-Path -Path (Get-Location) -ChildPath $OutDir
}

New-Item -ItemType Directory -Force -Path $outRootAbs | Out-Null

$files = Get-ChildItem -LiteralPath $inputRoot -Recurse -File | Where-Object { [string]::IsNullOrEmpty($_.Extension) }
if ($files.Count -eq 0) {
    throw "No extensionless files found under $inputRoot"
}

$details = foreach ($f in $files) {
    $bytes = [System.IO.File]::ReadAllBytes($f.FullName)
    $n = [Math]::Min($HeaderBytes, $bytes.Length)
    $slice = if ($n -gt 0) { $bytes[0..($n - 1)] } else { @() }
    $hex = if ($n -gt 0) { (($slice | ForEach-Object { "{0:X2}" -f $_ }) -join " ") } else { "" }
    $ascii = if ($n -gt 0) {
        -join ($slice | ForEach-Object {
            if ($_ -ge 32 -and $_ -le 126) { [char]$_ } else { "." }
        })
    } else { "" }

    [PSCustomObject]@{
        RelativePath = $f.FullName.Substring($inputRoot.Length).TrimStart('\')
        SizeBytes    = $f.Length
        HeaderHex    = $hex
        HeaderAscii  = $ascii
        LikelyType   = Get-LikelyType -Bytes $bytes
    }
}

$summary = $details |
    Group-Object LikelyType, HeaderHex |
    Sort-Object Count -Descending |
    ForEach-Object {
        $first = $_.Group | Select-Object -First 1
        [PSCustomObject]@{
            Count      = $_.Count
            LikelyType = $first.LikelyType
            HeaderHex  = $first.HeaderHex
            SamplePath = ($first.RelativePath)
        }
    }

$detailsPath = Join-Path -Path $outRootAbs -ChildPath "extensionless-signatures-details.csv"
$summaryPath = Join-Path -Path $outRootAbs -ChildPath "extensionless-signatures-summary.csv"

$details | Export-Csv -Path $detailsPath -NoTypeInformation -Encoding UTF8
$summary | Export-Csv -Path $summaryPath -NoTypeInformation -Encoding UTF8

Write-Host ("Scanned {0} extensionless files." -f $files.Count)
Write-Host ("Wrote details: {0}" -f $detailsPath)
Write-Host ("Wrote summary: {0}" -f $summaryPath)
