<#
.SYNOPSIS
    The measurements for the old-vs-new demonstration, unattended.

.DESCRIPTION
    Measurements only. Runs, in order, and logs everything to <Out>\transcript.txt:

      1. calibrate this array            ~18 min
      2. transient capture, one pixel    ~15 min   <- shows WHY the old method was wrong
      3. sweep, old method               ~16 min
      4. sweep, new method               1-3 h     (depends on the calibrated dwell)

    Analysis is deliberately NOT run here: it needs no hardware, it can be repeated and
    re-tuned afterwards, and a failing figure must never cost bench time. The exact
    analysis commands, with the run directories filled in, are printed at the end and
    written to <Out>\analyse.ps1 -- run that when the bench is free.

    A failed stage stops the script, because every later stage depends on it.
    Ctrl-C is safe at any point: the hardware parks itself.

.EXAMPLE
    .\run_comparison_demo.ps1 -Simulate
    Dry run against the built-in rig. No hardware, a couple of minutes. Do this first.

.EXAMPLE
    .\run_comparison_demo.ps1 -Array Array42_IQE -Pixel "35,36"
#>
[CmdletBinding()]
param(
    [string]$Array      = "Array42_IQE",
    [string]$Pixel      = "35,36",
    [string]$Roi        = "[32,32,63,63]",
    [string]$Out        = "demo",
    [double]$CalSeconds = 120,
    [string]$AnalysisRepo = "C:\_D\programs\SMILE_IQE_Analysis",
    [switch]$Simulate
)

# NOT "Stop". In Windows PowerShell 5.1 anything a native exe writes to stderr becomes
# an ErrorRecord, so with "Stop" a single numpy RuntimeWarning aborts the whole run --
# which it did, after the sweeps had already finished. Failures are detected explicitly
# from $LASTEXITCODE instead, and `throw` still stops the script regardless.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$outRoot = Join-Path $root $Out
New-Item -ItemType Directory -Force $outRoot | Out-Null
$log = Join-Path $outRoot "transcript.txt"
Start-Transcript -Path $log -Append | Out-Null

$sim = @()
if ($Simulate) { $sim = @("--simulate") }

function Say($msg) {
    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host ("  {0}   [{1}]" -f $msg, (Get-Date -Format "HH:mm:ss"))
    Write-Host ("=" * 70)
}

function Run-Measure($stage, $cmdArgs) {
    Say $stage
    Write-Host ("uv run python -m measure_iqe " + ($cmdArgs -join " "))
    & uv run python -m measure_iqe @cmdArgs
    if ($LASTEXITCODE -ne 0) {
        Stop-Transcript | Out-Null
        throw "$stage failed with exit code $LASTEXITCODE. Nothing after it can run; see $log"
    }
}

# Newest directory under a root — the run that was just created.
function Newest($path) {
    if (-not (Test-Path $path)) { return $null }
    $d = Get-ChildItem $path -Directory -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($d) { return $d.FullName }
    return $null
}

$started = Get-Date
Write-Host "array      : $Array"
Write-Host "pixel      : $Pixel"
Write-Host "ROI        : $Roi"
Write-Host "output     : $outRoot"
if ($Simulate) { Write-Host "MODE       : SIMULATE (no hardware)" }

# ---------------------------------------------------------------- 1. calibrate
$calDir = Join-Path $outRoot "cal"
Run-Measure "1/4  CALIBRATE $Array  (~18 min)" (@(
    "--calibrate", "--array", $Array, "--out", $calDir,
    "--set", "calibrate_seconds=$CalSeconds") + $sim)

$calFile = Join-Path $root "calibrations\$Array.yaml"
if (Test-Path $calFile) {
    Write-Host ""
    Write-Host "--- calibration result ---"
    # Plain string matching, printed with Write-Host: Select-String emits MatchInfo
    # objects that do not reliably reach the console through a redirected pipeline.
    $show = $false
    foreach ($line in (Get-Content $calFile)) {
        if ($line -match "^(recommended|never_settled|lower_bound|measured_utc|pixels):") { $show = $true }
        elseif ($line -match "^\S") { $show = $false }
        if ($show) { Write-Host ("    " + $line) }
    }
    if (Select-String -Path $calFile -Pattern "lower_bound: true" -Quiet) {
        Write-Warning ("Some captures never settled in ${CalSeconds}s, so the dwell is a LOWER BOUND. " +
                       "The demo still runs and the comparison is still valid - the new run's low-J " +
                       "points just carry more unrelaxed tail. Re-run with -CalSeconds 300 for a real number.")
    }
} else {
    Write-Warning "no calibration at $calFile - the sweep will use the defaults"
}

# ---------------------------------------------------------- 2. the transients
$tranDir = Join-Path $outRoot "transients"
Say "2/4  TRANSIENTS on pixel $Pixel  (~15 min)"
$tranArgs = @(
    "--what", "step", "--out", $tranDir, "--pixels", $Pixel,
    "--highc-both", "--ranges", "1e-6", "1e-5", "1e-4",
    "--voltages", "-0.80", "-0.85", "-1.00",
    "--nplc", "1.0", "--samples", "2000", "--v-start", "-1.2",
    "--max-presettle-s", "90", "--step-n-pm", "512")
if (-not $Simulate) {
    Write-Host ("uv run python .\characterize_capacitance.py " + ($tranArgs -join " "))
    & uv run python .\characterize_capacitance.py @tranArgs
    if ($LASTEXITCODE -ne 0) { Write-Warning "transient capture returned $LASTEXITCODE - continuing" }
} else {
    Write-Host "(skipped in -Simulate: characterize_capacitance.py needs real instruments)"
}
$tranRun = Newest $tranDir

# ------------------------------------------------------------- 3. old method
# --set sample_name, NOT --array: --array would apply the calibration and undo the
# legacy settle times, which is the one thing this run must keep.
$oldDir = Join-Path $outRoot "old"
Run-Measure "3/4  SWEEP, OLD METHOD  (~16 min)" (@(
    "--config", "configs\legacy_pre261014.yaml",
    "--set", "sample_name=$Array", "--set", "roi=$Roi", "--set", "n_fine=30",
    "--out", $oldDir) + $sim)
$oldRun = Newest $oldDir

# ------------------------------------------------------------- 4. new method
$newDir = Join-Path $outRoot "new"
Run-Measure "4/4  SWEEP, NEW METHOD  (1-3 h)" (@(
    "--array", $Array, "--set", "roi=$Roi", "--out", $newDir) + $sim)
$newRun = Newest $newDir

# ------------------------------- hand the analysis over as a ready-to-run script
$analyse = Join-Path $outRoot "analyse.ps1"
$lines = @(
    "# Analysis for the run of $($started.ToString('yyyy-MM-dd HH:mm')). No hardware needed.",
    "# Written by run_comparison_demo.ps1 with the run directories already filled in.",
    "`$ErrorActionPreference = 'Continue'",
    "Set-Location '$AnalysisRepo'")
if ($tranRun) {
    $lines += @("", "# WHY the old method was wrong", "python plot_transients.py '$tranRun'")
}
if ($oldRun -and $newRun) {
    $lines += @(
        "", "# WHAT it changes",
        "python analyze_array.py '$oldRun'",
        "python analyze_array.py '$newRun'",
        "python compare_runs.py '$oldRun' '$newRun' --label-old 'old method' --label-new 'new method'")
}
$lines | Set-Content -Path $analyse -Encoding utf8

Say ("MEASUREMENTS DONE in {0:hh\:mm\:ss}" -f ((Get-Date) - $started))
Write-Host "Nothing below needs the bench. Run the analysis with:"
Write-Host ""
Write-Host "    $analyse"
Write-Host ""
Write-Host "or by hand:"
Write-Host ""
foreach ($l in $lines) {
    if ($l -like "python *") { Write-Host "    $l" }
}
Write-Host ""
Write-Host "Then present, in this order:"
if ($tranRun) {
    Write-Host "  1. WHY it was wrong   $tranRun"
    Write-Host "       settling.png     the current crawls for seconds"
    Write-Host "       transients.png   the old 20 ms settle is off the left edge"
    Write-Host "       step.csv         High-C off: 42-52% of whatever range is forced"
}
if ($newRun) {
    Write-Host "  2. WHAT it changes    $newRun\comparison\"
    Write-Host "       eqe_vs_J.png     the headline"
    Write-Host "       iv.png           how much higher the old method reads"
    Write-Host "       leakage.png      why"
    Write-Host "       per_pixel.png    per pixel, not one average"
    Write-Host "       COMPARISON.md    the numbers"
}
Write-Host ""
Write-Host "  transcript: $log"
Stop-Transcript | Out-Null
