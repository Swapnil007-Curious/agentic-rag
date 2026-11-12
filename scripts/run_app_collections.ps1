<#
.SYNOPSIS
    Builds the app indexes and runs the hosted generator-scale ablation, unattended and resumably.

.DESCRIPTION
    PowerShell entry point for scripts/run_app_collections.py (steps, skip/resume logic, time
    estimates and logging live there, so both entry points behave identically):

      indian_law, human_biology, physics_chemistry: ingest -> chunk -> build_index (bge-small)
      cs_ai app index (+ Python Tutorial, Think Python): ingest -> chunk -> build_index
      consumer_law, cs_ai: eval_generation.py --split test --systems G2H G4H (hosted model)
      consumer_law, cs_ai: SUMMARY.md with the ablation section

    The ablation needs the API key in this shell: $env:LLM_API_KEY = "..." (it is read from the
    environment only and never written anywhere). Every finished step is skipped on the next
    run and an interrupted step resumes from its caches, so after a crash, a reboot or an
    exhausted API quota just run the same command again. The log is
    collections\app_pipeline_log.txt.

.PARAMETER DryRun
    Print the plan and the time estimate only.

.PARAMETER NoAblation
    Build the app indexes only (no API calls).

.PARAMETER WaitForQuota
    Wait for the API's daily token quota to renew instead of stopping (an unattended run that
    can take days on a free API account; keep the computer on).

.PARAMETER NoKeepAwake
    Allow the computer to sleep while the runner works.

.EXAMPLE
    .\scripts\run_app_collections.ps1
#>
param(
    [switch]$DryRun,
    [switch]$NoAblation,
    [switch]$WaitForQuota,
    [switch]$NoKeepAwake
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"   # fall back to the active environment
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:TOKENIZERS_PARALLELISM = "false"

$runnerArgs = @("scripts\run_app_collections.py")
if ($DryRun) { $runnerArgs += "--dry-run" }
if ($NoAblation) { $runnerArgs += "--no-ablation" }
if ($WaitForQuota) { $runnerArgs += "--wait-for-quota" }
if ($NoKeepAwake) { $runnerArgs += "--no-keep-awake" }

& $python @runnerArgs
$code = $LASTEXITCODE
if ($code -ne 0 -and -not $DryRun) {
    Write-Host ""
    Write-Host "The runner stopped (exit code $code). Fix the problem shown above, then run again:"
    $again = ".\scripts\run_app_collections.ps1"
    if ($NoAblation) { $again += " -NoAblation" }
    if ($WaitForQuota) { $again += " -WaitForQuota" }
    Write-Host "    $again"
}
exit $code
