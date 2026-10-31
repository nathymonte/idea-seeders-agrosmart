<#
.SYNOPSIS
    Executa localmente, via Docker, o mesmo pipeline DevSecOps do GitHub Actions.
.DESCRIPTION
    Etapas: 1) lint + testes + cobertura  2) SAST (Bandit) + dependencias (pip-audit)
            3) build da imagem  4) scan Trivy (Dockerfile e imagem)  5) smoke test do container.
    Relatorios em artifacts/ci/. Requer apenas o Docker Desktop aberto.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\ci_local.ps1
#>
param(
    [string]$Image = "agrosmart-dashboard:ci",
    [int]$SmokePort = 8599
)

# Continue: ferramentas nativas escrevem progresso no stderr; o resultado vem de $LASTEXITCODE
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Reports = Join-Path $Root "artifacts/ci"
New-Item -ItemType Directory -Force -Path $Reports | Out-Null
$Results = [ordered]@{}
$Start = Get-Date
$Mount = "${Root}:/src"

function Save-Utf8([string]$Path) {
    # Tee-Object do PowerShell 5.1 grava em UTF-16; aqui o relatorio sai em UTF-8 legivel
    begin { $lines = New-Object System.Collections.Generic.List[string] }
    process { $_; $lines.Add([string]$_) }
    end { [IO.File]::WriteAllLines($Path, $lines, (New-Object Text.UTF8Encoding($false))) }
}

function Invoke-Stage([string]$Name, [scriptblock]$Body) {
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    $watch = [Diagnostics.Stopwatch]::StartNew()
    & $Body
    $ok = ($LASTEXITCODE -eq 0)
    $Results[$Name] = @{ ok = $ok; seconds = [math]::Round($watch.Elapsed.TotalSeconds, 1) }
    if ($ok) { Write-Host "OK  $Name" -ForegroundColor Green } else { Write-Host "FALHOU  $Name" -ForegroundColor Red }
}

$PythonCi = @"
set -eo pipefail
cd /src
pip install -q --root-user-action=ignore -r requirements-dashboard.txt ruff coverage bandit pip-audit matplotlib==3.11.2
mkdir -p artifacts/ci
"@

Invoke-Stage "1. Qualidade: lint (Ruff) + testes + cobertura" {
    $cmd = "$PythonCi
ruff check . | tee artifacts/ci/ruff.txt
coverage run -m unittest discover -s tests -v 2>&1 | tee artifacts/ci/testes.txt
coverage report --fail-under=65 | tee artifacts/ci/cobertura.txt" -replace "`r", ""
    docker run --rm -v $Mount -e PYTHONDONTWRITEBYTECODE=1 -e AGROSMART_LOG_LEVEL=WARNING -e AGROSMART_LOG_DIR=/tmp/logs `
        -e COVERAGE_FILE=/tmp/.coverage python:3.11-slim bash -c $cmd
}

Invoke-Stage "2. Seguranca: SAST (Bandit) + dependencias (pip-audit)" {
    $cmd = "$PythonCi
bandit -r services src scripts dashboard -c pyproject.toml -f txt -o artifacts/ci/bandit.txt || true
bandit -q -r services src scripts dashboard -c pyproject.toml -ll
IGNORES=`$(grep -v '^#' security/pip-audit-ignore.txt | tr -d '\r' | sed '/^`$/d' | sed 's/^/--ignore-vuln /' | tr '\n' ' ')
pip-audit -r requirements-dashboard.txt `$IGNORES --progress-spinner off 2>&1 | tee artifacts/ci/pip-audit.txt" -replace "`r", ""
    docker run --rm -v $Mount python:3.11-slim bash -c $cmd
}

Invoke-Stage "3. Build da imagem Docker" {
    docker build --target dashboard -t $Image . 2>&1 | Save-Utf8 (Join-Path $Reports "docker-build.txt")
}

Invoke-Stage "4. Trivy: configuracao (Dockerfile) + vulnerabilidades da imagem" {
    $sock = "//var/run/docker.sock:/var/run/docker.sock"
    $cache = "agrosmart-trivy-cache:/root/.cache"
    docker run --rm -v "${Root}/Dockerfile:/cfg/Dockerfile:ro" -v $cache aquasec/trivy:latest `
        config --severity HIGH,CRITICAL --exit-code 1 /cfg | Save-Utf8 (Join-Path $Reports "trivy-config.txt")
    if ($LASTEXITCODE -ne 0) { return }
    docker run --rm -v $sock -v $cache -v "${Root}/.trivyignore:/src/.trivyignore:ro" aquasec/trivy:latest `
        image --severity MEDIUM,HIGH,CRITICAL --ignorefile /src/.trivyignore $Image | Save-Utf8 (Join-Path $Reports "trivy-imagem.txt")
    docker run --rm -v $sock -v $cache -v "${Root}/.trivyignore:/src/.trivyignore:ro" aquasec/trivy:latest `
        image --severity CRITICAL --ignore-unfixed --exit-code 1 --ignorefile /src/.trivyignore $Image
}

Invoke-Stage "5. Smoke test: container em execucao e saudavel" {
    docker rm -f agrosmart-smoke 2>$null | Out-Null
    docker run -d --name agrosmart-smoke -p "${SmokePort}:8501" $Image | Out-Null
    $status = ""
    for ($i = 1; $i -le 30; $i++) {
        $status = docker inspect --format "{{.State.Health.Status}}" agrosmart-smoke
        Write-Host "tentativa ${i}: $status"
        if ($status -eq "healthy") { break }
        Start-Sleep -Seconds 5
    }
    docker logs agrosmart-smoke 2>&1 | Select-Object -Last 15 | Save-Utf8 (Join-Path $Reports "smoke-logs.txt")
    docker rm -f agrosmart-smoke | Out-Null
    $global:LASTEXITCODE = if ($status -eq "healthy") { 0 } else { 1 }
}

Write-Host "`n=== Resumo do pipeline ($([math]::Round(((Get-Date) - $Start).TotalMinutes, 1)) min) ===" -ForegroundColor Cyan
$failed = $false
foreach ($stage in $Results.Keys) {
    $r = $Results[$stage]
    $label = if ($r.ok) { "OK     " } else { "FALHOU "; $failed = $true }
    Write-Host ("{0} {1,-60} {2,6}s" -f $label, $stage, $r.seconds) -ForegroundColor $(if ($r.ok) { "Green" } else { "Red" })
}
$Results | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $Reports "resumo.json")
Write-Host "Relatorios em $Reports"
if ($failed) { exit 1 }
