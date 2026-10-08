param([switch]$Demo, [int]$Port = 8501)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Crie .venv e instale requirements-dashboard.txt antes de iniciar.' }
if ($Demo) {
    $demoRoot = Join-Path $projectRoot '.demo'
    if (-not (Test-Path -LiteralPath (Join-Path $demoRoot 'summary.json'))) {
        Push-Location $projectRoot
        try { & $pythonPath -m scripts.demo; if ($LASTEXITCODE) { throw 'Falha ao gerar demonstracao.' } }
        finally { Pop-Location }
    }
    $env:DATA_LAKE_PATH = Join-Path $demoRoot 'data_lake'
    $env:AGROSMART_IMAGE_CSV = Join-Path $demoRoot 'output\classificacoes.csv'
}
& $pythonPath -m streamlit run (Join-Path $projectRoot 'dashboard\app.py') --server.port $Port --server.headless true
