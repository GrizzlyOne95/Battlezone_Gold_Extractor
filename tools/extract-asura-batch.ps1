param(
    [Parameter(Mandatory = $false)]
    [string]$GameRoot = "work/Battlezone",

    [Parameter(Mandatory = $false)]
    [string]$ExtractRoot = "work/extracted/full",

    [Parameter(Mandatory = $false)]
    [string]$LogsRoot = "work/logs/batch",

    [Parameter(Mandatory = $false)]
    [ValidateSet("Native", "QuickBMS", "Auto")]
    [string]$Extractor = "Native",

    [Parameter(Mandatory = $false)]
    [string]$PythonExe = "python",

    [Parameter(Mandatory = $false)]
    [string]$NativeExtractor = "tools\extract-asura-native.py",

    [Parameter(Mandatory = $false)]
    [string]$QuickBmsExe = "C:\Users\istuart\Downloads\quickbms_win\quickbms_4gb_files.exe",

    [Parameter(Mandatory = $false)]
    [string]$BmsScript = "tools\bms\asura.bms",

    [Parameter(Mandatory = $false)]
    [switch]$SkipExisting = $true,

    [Parameter(Mandatory = $false)]
    [ValidateSet("Legacy", "AllPcVariants")]
    [string]$TargetSet = "AllPcVariants"
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

function To-SafeArchiveName {
    param([string]$RelativePath)
    $name = $RelativePath -replace "[\\/]", "__"
    $name = $name -replace "\.", "_"
    return $name
}

function Resolve-WorkspacePathIfExists {
    param([string]$PathValue)
    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return $null
    }
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        if (Test-Path -LiteralPath $PathValue) {
            return (Resolve-Path -LiteralPath $PathValue).Path
        }
        return $null
    }
    $candidate = Join-Path -Path (Get-Location) -ChildPath $PathValue
    if (Test-Path -LiteralPath $candidate) {
        return (Resolve-Path -LiteralPath $candidate).Path
    }
    return $null
}

function Get-ExtractionTargets {
    param(
        [string]$Root,
        [string]$Mode
    )

    $targets = @()
    switch ($Mode) {
        "Legacy" {
            $charsPath = Join-Path -Path $Root -ChildPath "chars"
            if (Test-Path -LiteralPath $charsPath) {
                $targets += Get-ChildItem -LiteralPath $charsPath -Filter "*.pc" -File
            }
            $targets += Get-ChildItem -LiteralPath $Root -Recurse -Filter "*.pc_textures" -File
        }
        "AllPcVariants" {
            $targets += Get-ChildItem -LiteralPath $Root -Recurse -File |
                Where-Object { $_.Name -match "\.pc($|_)" }
        }
    }

    return @($targets | Sort-Object FullName -Unique)
}

function Test-AsuraPlaceholder {
    param([System.IO.FileInfo]$File)
    if ($File.Length -ne 12) {
        return $false
    }
    $bytes = [System.IO.File]::ReadAllBytes($File.FullName)
    if ($bytes.Length -ne 12) {
        return $false
    }
    $sig = [System.Text.Encoding]::ASCII.GetString($bytes, 0, 8)
    if ($sig -ne "Asura   ") {
        return $false
    }
    return ($bytes[8] -eq 0 -and $bytes[9] -eq 0 -and $bytes[10] -eq 0 -and $bytes[11] -eq 0)
}

function Get-StatusFromRun {
    param(
        [int]$ExitCode,
        [int]$OutputFiles,
        [string[]]$CommandOutput
    )

    if ($ExitCode -ne 0) {
        return "Failed"
    }

    if ($OutputFiles -gt 0) {
        return "OK"
    }

    $errorLine = $null
    if ($null -ne $CommandOutput -and $CommandOutput.Count -gt 0) {
        $errorLine = $CommandOutput |
            Where-Object { $_ -match "(?i)\berror:" } |
            Select-Object -First 1
    }

    if ($null -ne $errorLine) {
        return "Unsupported"
    }

    return "Empty"
}

function Invoke-CapturedCommand {
    param(
        [scriptblock]$CommandBlock
    )

    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = @(& $CommandBlock 2>&1 | ForEach-Object { "$_" })
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldEap
    }

    return [PSCustomObject]@{
        Output   = $output
        ExitCode = $exitCode
    }
}

$gameRoot = Resolve-WorkspacePath -PathValue $GameRoot

$nativeExtractorPath = Resolve-WorkspacePath -PathValue $NativeExtractor

$quickBms = $null
$bms = $null
if ($Extractor -eq "QuickBMS" -or $Extractor -eq "Auto") {
    $quickBms = Resolve-WorkspacePathIfExists -PathValue $QuickBmsExe
    $bms = Resolve-WorkspacePathIfExists -PathValue $BmsScript
    if ($null -eq $quickBms -or $null -eq $bms) {
        if ($Extractor -eq "QuickBMS") {
            throw "QuickBMS mode selected but QuickBMS executable or BMS script was not found."
        }
        Write-Host "Auto mode: QuickBMS fallback disabled (missing executable or BMS script)."
    }
}

$extractRootAbs = if ([System.IO.Path]::IsPathRooted($ExtractRoot)) {
    $ExtractRoot
} else {
    Join-Path -Path (Get-Location) -ChildPath $ExtractRoot
}
$logsRootAbs = if ([System.IO.Path]::IsPathRooted($LogsRoot)) {
    $LogsRoot
} else {
    Join-Path -Path (Get-Location) -ChildPath $LogsRoot
}

New-Item -ItemType Directory -Force -Path $extractRootAbs | Out-Null
New-Item -ItemType Directory -Force -Path $logsRootAbs | Out-Null

$targets = Get-ExtractionTargets -Root $gameRoot -Mode $TargetSet

if ($targets.Count -eq 0) {
    throw "No target archives found under $gameRoot"
}

Write-Host "Found $($targets.Count) archives to process (TargetSet=$TargetSet)."

$results = New-Object System.Collections.Generic.List[object]
$index = 0
foreach ($archive in $targets) {
    $index++
    $relative = $archive.FullName.Substring($gameRoot.Length).TrimStart('\')
    $safeName = To-SafeArchiveName -RelativePath $relative
    $outDir = Join-Path -Path $extractRootAbs -ChildPath $safeName
    $logFile = Join-Path -Path $logsRootAbs -ChildPath ($safeName + ".log")

    if (Test-AsuraPlaceholder -File $archive) {
        Write-Host ("[{0}/{1}] PLACEHOLDER {2}" -f $index, $targets.Count, $relative)
        $results.Add([PSCustomObject]@{
            RelativeArchive = $relative
            OutputDir       = $outDir
            Extractor       = $Extractor
            Status          = "Placeholder"
            ExitCode        = 0
            OutputFiles     = 0
            LogFile         = $logFile
        })
        continue
    }

    if ($SkipExisting -and (Test-Path -LiteralPath $outDir)) {
        $existingFiles = (Get-ChildItem -LiteralPath $outDir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
        if ($existingFiles -gt 0) {
            Write-Host ("[{0}/{1}] SKIP {2} ({3} files already)" -f $index, $targets.Count, $relative, $existingFiles)
            $results.Add([PSCustomObject]@{
                RelativeArchive = $relative
                OutputDir       = $outDir
                Extractor       = $Extractor
                Status          = "Skipped"
                ExitCode        = 0
                OutputFiles     = $existingFiles
                LogFile         = $logFile
            })
            continue
        }
    }

    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    Write-Host ("[{0}/{1}] EXTRACT {2} (Extractor={3})" -f $index, $targets.Count, $relative, $Extractor)

    $extractorUsed = $Extractor
    $status = "Failed"
    $exit = 1
    $cmdOutput = @()

    if ($Extractor -eq "QuickBMS") {
        $run = Invoke-CapturedCommand -CommandBlock { & $quickBms -Y $bms $archive.FullName $outDir }
        $cmdOutput = $run.Output
        $cmdOutput | Tee-Object -FilePath $logFile | Out-Null
        $exit = $run.ExitCode
        $outFiles = (Get-ChildItem -LiteralPath $outDir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
        $status = Get-StatusFromRun -ExitCode $exit -OutputFiles $outFiles -CommandOutput $cmdOutput
    }
    else {
        if ($nativeExtractorPath.ToLowerInvariant().EndsWith(".exe")) {
            $run = Invoke-CapturedCommand -CommandBlock { & $nativeExtractorPath --input $archive.FullName --output-dir $outDir }
        }
        else {
            if ([string]::IsNullOrWhiteSpace($PythonExe)) {
                throw "PythonExe is required when NativeExtractor is a .py script."
            }
            $run = Invoke-CapturedCommand -CommandBlock { & $PythonExe $nativeExtractorPath --input $archive.FullName --output-dir $outDir }
        }
        $cmdOutput = $run.Output
        $cmdOutput | Tee-Object -FilePath $logFile | Out-Null
        $exit = $run.ExitCode
        $outFiles = (Get-ChildItem -LiteralPath $outDir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count

        if ($exit -eq 0) {
            if ($outFiles -gt 0) {
                $status = "OK"
            }
            else {
                $status = "Empty"
            }
        }
        elseif ($exit -eq 10) {
            $status = "Unsupported"
        }
        else {
            $status = "Failed"
        }

        if ($Extractor -eq "Auto" -and $status -eq "Unsupported" -and $null -ne $quickBms -and $null -ne $bms) {
            Write-Host ("[{0}/{1}] FALLBACK QuickBMS {2}" -f $index, $targets.Count, $relative)
            $extractorUsed = "Auto(QuickBMSFallback)"
            $run = Invoke-CapturedCommand -CommandBlock { & $quickBms -Y $bms $archive.FullName $outDir }
            $cmdOutput = $run.Output
            $cmdOutput | Tee-Object -FilePath $logFile -Append | Out-Null
            $exit = $run.ExitCode
            $outFiles = (Get-ChildItem -LiteralPath $outDir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
            $status = Get-StatusFromRun -ExitCode $exit -OutputFiles $outFiles -CommandOutput $cmdOutput
        }
    }

    $outFiles = (Get-ChildItem -LiteralPath $outDir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object).Count
    $results.Add([PSCustomObject]@{
        RelativeArchive = $relative
        OutputDir       = $outDir
        Extractor       = $extractorUsed
        Status          = $status
        ExitCode        = $exit
        OutputFiles     = $outFiles
        LogFile         = $logFile
    })
}

$reportPath = Join-Path -Path $logsRootAbs -ChildPath "extract-report.csv"
$results | Export-Csv -Path $reportPath -NoTypeInformation -Encoding UTF8

$ok = @($results | Where-Object { $_.Status -eq "OK" }).Count
$skipped = @($results | Where-Object { $_.Status -eq "Skipped" }).Count
$placeholders = @($results | Where-Object { $_.Status -eq "Placeholder" }).Count
$unsupported = @($results | Where-Object { $_.Status -eq "Unsupported" }).Count
$empty = @($results | Where-Object { $_.Status -eq "Empty" }).Count
$failed = @($results | Where-Object { $_.Status -eq "Failed" }).Count

Write-Host ("Done. OK={0}, Skipped={1}, Placeholder={2}, Unsupported={3}, Empty={4}, Failed={5}" -f $ok, $skipped, $placeholders, $unsupported, $empty, $failed)
Write-Host ("Report: {0}" -f $reportPath)

if ($failed -gt 0) {
    exit 2
}
