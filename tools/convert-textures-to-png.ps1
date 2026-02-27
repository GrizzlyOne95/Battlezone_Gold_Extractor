param(
    [Parameter(Mandatory = $false)]
    [string]$InputRoot = "work/extracted/full",

    [Parameter(Mandatory = $false)]
    [string]$OutputRoot = "work/preview/png",

    [Parameter(Mandatory = $false)]
    [string]$ReportPath = "notes/texture-conversion-report.csv",

    [Parameter(Mandatory = $false)]
    [switch]$SkipExisting = $true,

    [Parameter(Mandatory = $false)]
    [string]$TexconvExe = "",

    [Parameter(Mandatory = $false)]
    [switch]$UseTexconvFallback = $true,

    [Parameter(Mandatory = $false)]
    [switch]$PreferTexconv
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-WorkspacePathOrCreate {
    param(
        [string]$PathValue,
        [switch]$CreateDirectory
    )

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        if ($CreateDirectory -and -not (Test-Path -LiteralPath $PathValue)) {
            New-Item -ItemType Directory -Force -Path $PathValue | Out-Null
        }
        return $PathValue
    }

    $candidate = Join-Path -Path (Get-Location) -ChildPath $PathValue
    if ($CreateDirectory -and -not (Test-Path -LiteralPath $candidate)) {
        New-Item -ItemType Directory -Force -Path $candidate | Out-Null
    }
    return $candidate
}

function Resolve-TexconvPath {
    param([string]$ExplicitPath)

    if (-not [string]::IsNullOrWhiteSpace($ExplicitPath)) {
        $resolvedExplicit = Resolve-WorkspacePathOrCreate -PathValue $ExplicitPath
        if (-not (Test-Path -LiteralPath $resolvedExplicit)) {
            throw "TexconvExe does not exist: $resolvedExplicit"
        }
        return $resolvedExplicit
    }

    $cmd = Get-Command texconv -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $candidates = @(
        (Join-Path -Path (Get-Location) -ChildPath "tools\bin\texconv.exe"),
        "C:\Program Files\Microsoft DirectXTex\texconv.exe",
        "C:\Program Files (x86)\Microsoft DirectXTex\texconv.exe",
        "C:\Tools\DirectXTex\texconv.exe",
        "C:\tools\DirectXTex\texconv.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    return $null
}

function Get-DdsHeaderInfo {
    param([string]$Path)

    $bytes = [System.IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -lt 128) {
        return [PSCustomObject]@{
            IsDds  = $false
            FourCC = ""
            DXGI   = ""
        }
    }

    $magic = [System.Text.Encoding]::ASCII.GetString($bytes, 0, 4)
    if ($magic -ne "DDS ") {
        return [PSCustomObject]@{
            IsDds  = $false
            FourCC = ""
            DXGI   = ""
        }
    }

    $fourCC = [System.Text.Encoding]::ASCII.GetString($bytes, 84, 4)
    $dxgi = ""
    if ($fourCC -eq "DX10" -and $bytes.Length -ge 132) {
        $dxgi = [BitConverter]::ToUInt32($bytes, 128).ToString()
    }

    return [PSCustomObject]@{
        IsDds  = $true
        FourCC = $fourCC
        DXGI   = $dxgi
    }
}

function Convert-WithFfmpeg {
    param(
        [string]$FfmpegPath,
        [string]$SourcePath,
        [string]$DestPath
    )

    & $FfmpegPath -hide_banner -loglevel error -y -i $SourcePath $DestPath
    return $LASTEXITCODE
}

function Convert-WithTexconv {
    param(
        [string]$TexconvPath,
        [string]$SourcePath,
        [string]$DestPath
    )

    if (-not $TexconvPath) {
        return 127
    }

    $destDir = Split-Path -Path $DestPath -Parent
    $texconvInput = $SourcePath
    $tempInput = $null

    try {
        if ([System.IO.Path]::GetExtension($SourcePath).ToLowerInvariant() -ne ".dds") {
            $tempInput = Join-Path -Path ([System.IO.Path]::GetTempPath()) -ChildPath ("bz_texconv_" + [Guid]::NewGuid().ToString("N") + ".dds")
            Copy-Item -LiteralPath $SourcePath -Destination $tempInput -Force
            $texconvInput = $tempInput
        }

        $sourceName = [System.IO.Path]::GetFileNameWithoutExtension($texconvInput)
        & $TexconvPath -nologo -y -ft png -o $destDir $texconvInput | Out-Null
        $exit = $LASTEXITCODE
        if ($exit -ne 0) {
            return $exit
        }

        $texconvOutput = Join-Path -Path $destDir -ChildPath ($sourceName + ".png")
        if (Test-Path -LiteralPath $texconvOutput) {
            if ($texconvOutput -ne $DestPath) {
                Move-Item -LiteralPath $texconvOutput -Destination $DestPath -Force
            }
            return 0
        }

        $matches = Get-ChildItem -LiteralPath $destDir -File -Filter ($sourceName + "*.png") -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending
        if ($matches.Count -gt 0) {
            Move-Item -LiteralPath $matches[0].FullName -Destination $DestPath -Force
            return 0
        }

        return 3
    }
    finally {
        if ($tempInput -and (Test-Path -LiteralPath $tempInput)) {
            Remove-Item -LiteralPath $tempInput -Force -ErrorAction SilentlyContinue
        }
    }
}

$inputAbs = Resolve-WorkspacePathOrCreate -PathValue $InputRoot
if (-not (Test-Path -LiteralPath $inputAbs)) {
    throw "InputRoot does not exist: $inputAbs"
}

$outputAbs = Resolve-WorkspacePathOrCreate -PathValue $OutputRoot -CreateDirectory
$reportAbs = if ([System.IO.Path]::IsPathRooted($ReportPath)) {
    $ReportPath
} else {
    Join-Path -Path (Get-Location) -ChildPath $ReportPath
}
$reportDir = Split-Path -Path $reportAbs -Parent
if (-not (Test-Path -LiteralPath $reportDir)) {
    New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
}

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    throw "ffmpeg not found in PATH."
}

$texconvPath = Resolve-TexconvPath -ExplicitPath $TexconvExe
$texconvEnabled = ($null -ne $texconvPath)
if ($UseTexconvFallback -and -not $texconvEnabled) {
    Write-Host "texconv not found; fallback disabled."
}

$targets = Get-ChildItem -LiteralPath $inputAbs -Recurse -File |
    Where-Object { $_.Extension.ToLowerInvariant() -in @(".tga", ".dds", ".bmp") }

if ($targets.Count -eq 0) {
    throw "No texture files found under $inputAbs"
}

Write-Host ("Converting {0} files (ffmpeg + optional texconv fallback)..." -f $targets.Count)

$results = New-Object System.Collections.Generic.List[object]
$i = 0
foreach ($src in $targets) {
    $i++
    $relative = $src.FullName.Substring($inputAbs.Length).TrimStart('\')
    $pngRelative = [System.IO.Path]::ChangeExtension($relative, ".png")
    $dst = Join-Path -Path $outputAbs -ChildPath $pngRelative
    $dstDir = Split-Path -Path $dst -Parent
    if (-not (Test-Path -LiteralPath $dstDir)) {
        New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
    }

    if ($SkipExisting -and (Test-Path -LiteralPath $dst)) {
        Write-Host ("[{0}/{1}] SKIP {2}" -f $i, $targets.Count, $relative)
        $results.Add([PSCustomObject]@{
            SourcePath = $relative
            DestPath   = $pngRelative
            Status     = "Skipped"
            ExitCode   = 0
            Converter  = "Skipped"
            IsDDS      = ""
            FourCC     = ""
            DXGI       = ""
        })
        continue
    }

    $ddsInfo = Get-DdsHeaderInfo -Path $src.FullName
    Write-Host ("[{0}/{1}] CONVERT {2}" -f $i, $targets.Count, $relative)

    $converterUsed = ""
    $exit = 1
    $attemptedTexconv = $false

    if ($PreferTexconv -and $texconvEnabled -and $ddsInfo.IsDds) {
        $exit = Convert-WithTexconv -TexconvPath $texconvPath -SourcePath $src.FullName -DestPath $dst
        $attemptedTexconv = $true
        if ($exit -eq 0) {
            $converterUsed = "texconv"
        }
    }

    if ($exit -ne 0) {
        $exit = Convert-WithFfmpeg -FfmpegPath $ffmpeg.Source -SourcePath $src.FullName -DestPath $dst
        if ($exit -eq 0) {
            $converterUsed = "ffmpeg"
        }
    }

    if (
        $exit -ne 0 -and
        $UseTexconvFallback -and
        $texconvEnabled -and
        $ddsInfo.IsDds -and
        -not $attemptedTexconv
    ) {
        $exit = Convert-WithTexconv -TexconvPath $texconvPath -SourcePath $src.FullName -DestPath $dst
        if ($exit -eq 0) {
            $converterUsed = "texconv"
        }
    }

    $status = if ($exit -eq 0) { "OK" } else { "Failed" }
    if ([string]::IsNullOrEmpty($converterUsed)) {
        $converterUsed = "none"
    }

    $results.Add([PSCustomObject]@{
        SourcePath = $relative
        DestPath   = $pngRelative
        Status     = $status
        ExitCode   = $exit
        Converter  = $converterUsed
        IsDDS      = $ddsInfo.IsDds
        FourCC     = $ddsInfo.FourCC
        DXGI       = $ddsInfo.DXGI
    })
}

$results | Export-Csv -Path $reportAbs -NoTypeInformation -Encoding UTF8

$ok = @($results | Where-Object { $_.Status -eq "OK" }).Count
$skipped = @($results | Where-Object { $_.Status -eq "Skipped" }).Count
$failed = @($results | Where-Object { $_.Status -eq "Failed" }).Count

Write-Host ("Done. OK={0}, Skipped={1}, Failed={2}" -f $ok, $skipped, $failed)
Write-Host ("Report: {0}" -f $reportAbs)

if ($failed -gt 0) {
    exit 2
}
