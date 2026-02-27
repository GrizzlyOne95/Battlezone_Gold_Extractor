param(
    [Parameter(Mandatory = $false)]
    [string]$PythonExe = "python",

    [Parameter(Mandatory = $false)]
    [string]$DistDir = "dist",

    [Parameter(Mandatory = $false)]
    [string]$BuildDir = "build/pyinstaller",

    [Parameter(Mandatory = $false)]
    [switch]$OneFile = $true,

    [Parameter(Mandatory = $false)]
    [switch]$Windowed = $true,

    [Parameter(Mandatory = $false)]
    [switch]$BundleRuntimeBin = $true,

    [Parameter(Mandatory = $false)]
    [switch]$BundleBms = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-PathFromBase {
    param(
        [string]$PathValue,
        [string]$BasePath
    )
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
    return Join-Path -Path $BasePath -ChildPath $PathValue
}

function Invoke-NativeProcess {
    param(
        [string]$Executable,
        [string[]]$Arguments = @()
    )

    $stdoutFile = [System.IO.Path]::GetTempFileName()
    $stderrFile = [System.IO.Path]::GetTempFileName()

    try {
        $proc = Start-Process `
            -FilePath $Executable `
            -ArgumentList $Arguments `
            -Wait `
            -PassThru `
            -NoNewWindow `
            -RedirectStandardOutput $stdoutFile `
            -RedirectStandardError $stderrFile

        if (Test-Path -LiteralPath $stdoutFile) {
            Get-Content -LiteralPath $stdoutFile | ForEach-Object { Write-Host $_ }
        }
        if (Test-Path -LiteralPath $stderrFile) {
            Get-Content -LiteralPath $stderrFile | ForEach-Object { Write-Host $_ }
        }
        return $proc.ExitCode
    }
    finally {
        if (Test-Path -LiteralPath $stdoutFile) { Remove-Item -LiteralPath $stdoutFile -Force -ErrorAction SilentlyContinue }
        if (Test-Path -LiteralPath $stderrFile) { Remove-Item -LiteralPath $stderrFile -Force -ErrorAction SilentlyContinue }
    }
}

function Invoke-PyInstallerBuild {
    param(
        [string]$EntryScript,
        [string]$Name,
        [switch]$WindowedBuild,
        [switch]$OneFileBuild,
        [string]$DistPath,
        [string]$BuildPath,
        [string[]]$ExtraArgs = @()
    )

    $args = @(
        "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name", $Name,
        "--distpath", $DistPath,
        "--workpath", $BuildPath,
        "--specpath", $BuildPath
    )

    if ($OneFileBuild) {
        $args += "--onefile"
    }
    else {
        $args += "--onedir"
    }

    if ($WindowedBuild) {
        $args += "--windowed"
    }
    else {
        $args += "--console"
    }

    if ($ExtraArgs -and $ExtraArgs.Count -gt 0) {
        $args += $ExtraArgs
    }

    $args += $EntryScript

    Write-Host ("Building {0} from {1}..." -f $Name, $EntryScript)
    $rc = Invoke-NativeProcess -Executable $PythonExe -Arguments $args
    if ($rc -ne 0) {
        throw "PyInstaller failed for $Name (exit $rc)"
    }
}

function New-PyInstallerBundleArg {
    param(
        [string]$SourcePath,
        [string]$DestPath
    )
    return ("{0};{1}" -f $SourcePath, $DestPath)
}

$scriptRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$repoRoot = (Resolve-Path -LiteralPath (Join-Path -Path $scriptRoot -ChildPath "..")).Path
$toolsRoot = $scriptRoot
$distAbs = Resolve-PathFromBase -PathValue $DistDir -BasePath $repoRoot
$buildAbs = Resolve-PathFromBase -PathValue $BuildDir -BasePath $repoRoot

if (-not (Test-Path -LiteralPath $distAbs)) {
    New-Item -ItemType Directory -Force -Path $distAbs | Out-Null
}
if (-not (Test-Path -LiteralPath $buildAbs)) {
    New-Item -ItemType Directory -Force -Path $buildAbs | Out-Null
}

# Ensure PyInstaller is available.
$pyiVersionRc = Invoke-NativeProcess -Executable $PythonExe -Arguments @("-m", "PyInstaller", "--version")
if ($pyiVersionRc -ne 0) {
    throw "PyInstaller not found for '$PythonExe'. Install with: pip install pyinstaller"
}

# Build helper console executables (always onefile).
Invoke-PyInstallerBuild -EntryScript (Join-Path $toolsRoot "bzg-extractor.py") -Name "bzg-extractor" -DistPath $distAbs -BuildPath $buildAbs -WindowedBuild:$false -OneFileBuild
Invoke-PyInstallerBuild -EntryScript (Join-Path $toolsRoot "extract-asura-native.py") -Name "extract-asura-native" -DistPath $distAbs -BuildPath $buildAbs -WindowedBuild:$false -OneFileBuild
Invoke-PyInstallerBuild -EntryScript (Join-Path $toolsRoot "export-hskn-obj-candidates.py") -Name "export-hskn-obj-candidates" -DistPath $distAbs -BuildPath $buildAbs -WindowedBuild:$false -OneFileBuild
Invoke-PyInstallerBuild -EntryScript (Join-Path $toolsRoot "export-model-obj-candidates.py") -Name "export-model-obj-candidates" -DistPath $distAbs -BuildPath $buildAbs -WindowedBuild:$false -OneFileBuild

$helperExes = @(
    "bzg-extractor.exe",
    "extract-asura-native.exe",
    "export-hskn-obj-candidates.exe",
    "export-model-obj-candidates.exe"
)

$uiExtraArgs = @()
if ($OneFile) {
    foreach ($exeName in $helperExes) {
        $src = Join-Path -Path $distAbs -ChildPath $exeName
        if (-not (Test-Path -LiteralPath $src)) {
            throw "Missing helper exe for embedding: $src"
        }
        $uiExtraArgs += "--add-binary"
        $uiExtraArgs += (New-PyInstallerBundleArg -SourcePath $src -DestPath ".")
    }

    $runtimeScripts = @(
        "extract-asura-batch.ps1",
        "convert-textures-to-png.ps1",
        "cleanup-workspace.ps1"
    )
    foreach ($s in $runtimeScripts) {
        $src = Join-Path $toolsRoot $s
        if (-not (Test-Path -LiteralPath $src)) {
            throw "Missing runtime script for embedding: $src"
        }
        $uiExtraArgs += "--add-data"
        $uiExtraArgs += (New-PyInstallerBundleArg -SourcePath $src -DestPath "tools")
    }

    if ($BundleBms) {
        $bmsSrc = Join-Path -Path $toolsRoot -ChildPath "bms"
        if (Test-Path -LiteralPath $bmsSrc) {
            $uiExtraArgs += "--add-data"
            $uiExtraArgs += (New-PyInstallerBundleArg -SourcePath $bmsSrc -DestPath "tools/bms")
        }
    }

    if ($BundleRuntimeBin) {
        $binSrc = Join-Path -Path $toolsRoot -ChildPath "bin"
        if (Test-Path -LiteralPath $binSrc) {
            $uiExtraArgs += "--add-data"
            $uiExtraArgs += (New-PyInstallerBundleArg -SourcePath $binSrc -DestPath "tools/bin")
        }
    }
}

# Build UI executable.
Invoke-PyInstallerBuild `
    -EntryScript (Join-Path $toolsRoot "bzg-extractor-ui.py") `
    -Name "BattlezoneGoldExtractor" `
    -DistPath $distAbs `
    -BuildPath $buildAbs `
    -WindowedBuild:$Windowed `
    -OneFileBuild:$OneFile `
    -ExtraArgs $uiExtraArgs

$bundleRoot = if ($OneFile) {
    $distAbs
}
else {
    Join-Path -Path $distAbs -ChildPath "BattlezoneGoldExtractor"
}

if (-not (Test-Path -LiteralPath $bundleRoot)) {
    throw "Expected bundle root not found: $bundleRoot"
}

if (-not $OneFile) {
    # Ensure helper exes are placed beside the UI exe.
    foreach ($exeName in $helperExes) {
        $src = Join-Path -Path $distAbs -ChildPath $exeName
        if (-not (Test-Path -LiteralPath $src)) {
            throw "Missing helper exe after build: $src"
        }
        if ($bundleRoot -ne $distAbs) {
            Copy-Item -LiteralPath $src -Destination (Join-Path $bundleRoot $exeName) -Force
        }
    }

    $toolsOut = Join-Path -Path $bundleRoot -ChildPath "tools"
    if (-not (Test-Path -LiteralPath $toolsOut)) {
        New-Item -ItemType Directory -Force -Path $toolsOut | Out-Null
    }

    # Runtime scripts needed by backend.
    $runtimeScripts = @(
        "extract-asura-batch.ps1",
        "convert-textures-to-png.ps1",
        "cleanup-workspace.ps1"
    )
    foreach ($s in $runtimeScripts) {
        Copy-Item -LiteralPath (Join-Path $toolsRoot $s) -Destination (Join-Path $toolsOut $s) -Force
    }

    if ($BundleBms) {
        $bmsSrc = Join-Path -Path $toolsRoot -ChildPath "bms"
        if (Test-Path -LiteralPath $bmsSrc) {
            Copy-Item -LiteralPath $bmsSrc -Destination (Join-Path $toolsOut "bms") -Recurse -Force
        }
    }

    if ($BundleRuntimeBin) {
        $binSrc = Join-Path -Path $toolsRoot -ChildPath "bin"
        if (Test-Path -LiteralPath $binSrc) {
            Copy-Item -LiteralPath $binSrc -Destination (Join-Path $toolsOut "bin") -Recurse -Force
        }
    }
}
else {
    # Keep one-file output clean: helper exes are embedded into the UI binary.
    foreach ($exeName in $helperExes) {
        $src = Join-Path -Path $distAbs -ChildPath $exeName
        if (Test-Path -LiteralPath $src) {
            Remove-Item -LiteralPath $src -Force
        }
    }
    Write-Host "Onefile mode: runtime helpers/resources embedded into BattlezoneGoldExtractor.exe"
}

Write-Host ("Build complete: {0}" -f $bundleRoot)
