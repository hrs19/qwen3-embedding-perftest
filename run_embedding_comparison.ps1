# Runs the full Qwen3-Embedding generation comparison matrix (model size x device x output
# dimension) and produces the Excel report. Mirrors the runs used to produce comparison.xlsx.
#
# Usage (from a PowerShell prompt, any working directory):
#   .\run_embedding_comparison.ps1
#   .\run_embedding_comparison.ps1 -NumSamples 1000
#   .\run_embedding_comparison.ps1 -Models 0.6b,4b -Devices cuda

param(
    [string[]]$Models = @("0.6b", "4b", "8b"),
    [string[]]$Devices = @("cpu", "cuda"),
    [string[]]$Dimensions = @("768", "1024", "default"),
    [int]$NumSamples = 500,
    [int]$MaxLength = 256,
    [int]$Seed = 42
)

$ErrorActionPreference = "Continue"  # keep going even if one run fails (e.g. RAM guardrail)

# Per-model batch size (kept small for larger/quantized models to fit 8GB VRAM)
$BatchSizes = @{ "0.6b" = 16; "4b" = 8; "8b" = 4 }

$RepoRoot = $PSScriptRoot
$Python = "C:\Users\harsh\anaconda3\envs\qwen3-embed\python.exe"

# Safety net: the qwen3-embed conda env's SSL_CERT_FILE points at a real cert bundle;
# set it explicitly too in case this script is run without `conda activate`.
$env:SSL_CERT_FILE = "C:\Users\harsh\anaconda3\envs\qwen3-embed\Library\ssl\cacert.pem"

Set-Location "$RepoRoot\src"

$total = $Models.Count * $Devices.Count * $Dimensions.Count
$i = 0

foreach ($model in $Models) {
    $bs = $BatchSizes[$model]
    foreach ($device in $Devices) {
        foreach ($dim in $Dimensions) {
            $i++
            Write-Host ""
            Write-Host "=== [$i/$total] model=$model device=$device dimension=$dim batch_size=$bs ===" -ForegroundColor Cyan
            & $Python generate_embeddings.py `
                --model-size $model `
                --device $device `
                --dimension $dim `
                --num-samples $NumSamples `
                --batch-size $bs `
                --max-length $MaxLength `
                --seed $Seed
        }
    }
}

Set-Location $RepoRoot
Write-Host ""
Write-Host "=== All runs done. Generating comparison report... ===" -ForegroundColor Cyan
& $Python compare_embedding_results.py

Write-Host ""
Write-Host "Excel report: $RepoRoot\results\embeddings\comparison.xlsx" -ForegroundColor Green
