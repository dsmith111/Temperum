<#
.SYNOPSIS
    Publish a model: convert → pack → ollama create → ollama push

.EXAMPLE
    .\scripts\publish_model.ps1 `
        -Input "C:\models\phi3-mini-onnx-qnn" `
        -Name "dsmith111/phi3:mini-qnn" `
        -Mode "prebuilt-ort-genai" `
        -OutDir ".\out\phi3-mini-qnn"

.EXAMPLE
    .\scripts\publish_model.ps1 `
        -Input "hf:microsoft/Phi-3-mini-4k-instruct" `
        -Name "dsmith111/phi3:mini-qnn" `
        -Mode "export" `
        -OutDir ".\out\phi3-mini-qnn"
#>
param(
    [Parameter(Mandatory)][string]$Input,
    [Parameter(Mandatory)][string]$Name,
    [Parameter(Mandatory)][string]$OutDir,
    [string]$Mode = "prebuilt-ort-genai",
    [string]$Backend = "qnn",
    [string]$Target = "auto",
    [string]$CompileStrategy = "passthrough",
    [int]$NumCtx = 512,
    [int]$NumPredict = 128,
    [switch]$NoPush,
    [switch]$KeepWork
)

$ErrorActionPreference = "Stop"

Write-Host "`n=== npu-model publish pipeline ===" -ForegroundColor Cyan
Write-Host "Input:    $Input"
Write-Host "Name:     $Name"
Write-Host "Mode:     $Mode"
Write-Host "Backend:  $Backend"
Write-Host "Target:   $Target"
Write-Host "Output:   $OutDir"
Write-Host ""

# Step 1: Doctor check
Write-Host "--- Step 1: Preflight check ---" -ForegroundColor Yellow
npu-model doctor
if ($LASTEXITCODE -ne 0) {
    Write-Host "Doctor check failed. Fix issues above before continuing." -ForegroundColor Red
    exit 1
}

# Step 2: Convert + Pack
Write-Host "`n--- Step 2: Convert + Pack ---" -ForegroundColor Yellow
$convertArgs = @(
    "convert",
    "--input", $Input,
    "--out", $OutDir,
    "--backend", $Backend,
    "--target", $Target,
    "--mode", $Mode,
    "--compile-strategy", $CompileStrategy,
    "--pack-ollama", $Name,
    "--num-ctx", $NumCtx,
    "--num-predict", $NumPredict
)
if ($KeepWork) {
    $convertArgs += "--keep-work"
}
npu-model @convertArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Conversion failed." -ForegroundColor Red
    exit 1
}

# Step 3: Validate
$publishDir = Join-Path $OutDir "ollama_publish"
Write-Host "`n--- Step 3: Validate ---" -ForegroundColor Yellow
npu-model validate --input $publishDir --as ollama-ortgenai
if ($LASTEXITCODE -ne 0) {
    Write-Host "Validation failed." -ForegroundColor Red
    exit 1
}

# Step 4: Ollama create
Write-Host "`n--- Step 4: ollama create ---" -ForegroundColor Yellow
Push-Location $publishDir
try {
    ollama create $Name -f Modelfile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ollama create failed." -ForegroundColor Red
        exit 1
    }
} finally {
    Pop-Location
}

# Step 5: Ollama push (optional)
if (-not $NoPush) {
    Write-Host "`n--- Step 5: ollama push ---" -ForegroundColor Yellow
    ollama push $Name
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ollama push failed." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n--- Step 5: ollama push (skipped: -NoPush) ---" -ForegroundColor Yellow
}

Write-Host "`n=== Done ===" -ForegroundColor Green
if ($NoPush) {
    Write-Host "Model installed locally as: $Name (no remote upload)"
} else {
    Write-Host "Model installed locally and pushed remotely as: $Name"
}
Write-Host "Generation and native execution have not been validated by this script."
Write-Host "Run with: ollama run $Name"
