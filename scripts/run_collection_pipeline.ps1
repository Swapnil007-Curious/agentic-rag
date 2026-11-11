<#
.SYNOPSIS
    Runs a collection's whole pipeline unattended (overnight), resumably.

.DESCRIPTION
    PowerShell entry point for scripts/run_collection_pipeline.py (the steps, skip/resume logic,
    time estimates, logging with peak RSS and SUMMARY.md live there, so both entry points behave
    identically):

      ingest -> chunk -> build_index -> build_auto_eval -> validate_eval (stops on failure) ->
      ollama stop -> retrieval dev grid -> select_configs -> eval_retrieval test ->
      run_generation dev -> derive_systems dev -> eval_generation test -> SUMMARY.md

    Every finished step is skipped on the next run, and an interrupted step resumes from its own
    caches, so after a crash or a reboot just run the same command again. The log is
    <results_dir>\pipeline_log.txt (collections\cs_ai\results\pipeline_log.txt for cs_ai).

.PARAMETER Collection
    Collection name from configs\collections.yaml (default: cs_ai).

.PARAMETER DryRun
    Print the plan and the time estimate only.

.PARAMETER NoKeepAwake
    Allow the computer to sleep while the pipeline runs.

.EXAMPLE
    .\scripts\run_collection_pipeline.ps1 -Collection cs_ai
#>
param(
    [string]$Collection = "cs_ai",
    [switch]$DryRun,
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

$runnerArgs = @("scripts\run_collection_pipeline.py", "--collection", $Collection)
if ($DryRun) { $runnerArgs += "--dry-run" }
if ($NoKeepAwake) { $runnerArgs += "--no-keep-awake" }

& $python @runnerArgs
$code = $LASTEXITCODE
if ($code -ne 0 -and -not $DryRun) {
    Write-Host ""
    Write-Host "The pipeline stopped (exit code $code). Fix the problem shown above, then run again:"
    Write-Host "    .\scripts\run_collection_pipeline.ps1 -Collection $Collection"
}
exit $code
