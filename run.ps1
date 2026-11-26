# Runs PulseIQ8_AI on Windows without Docker (uses Ollama on localhost:11434: the Ollama app or "docker compose up -d ollama").
# Same as run.sh on macOS: creates .venv, installs requirements, loads .env and starts the service (default port 8091).
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path .env)) { Copy-Item .env.example .env }

# Python 3.9+ (the py launcher first, then python on PATH; the Microsoft Store stub is skipped)
$py = $null
foreach ($c in @(@('py', '-3.12'), @('py', '-3.11'), @('py', '-3'), @('python'))) {
    if (-not (Get-Command $c[0] -ErrorAction SilentlyContinue)) { continue }
    $args2 = @($c | Select-Object -Skip 1)
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    $v = (& $c[0] @args2 -c 'import sys; print("%d.%d" % sys.version_info[:2]) if sys.version_info >= (3, 9) else print("old")' 2>$null)
    $ErrorActionPreference = $old
    if ($LASTEXITCODE -eq 0 -and $v -match '^\d+\.\d+$') { $py = $c; break }
}
if (-not $py) { Write-Host 'Python 3.9+ is needed: winget install Python.Python.3.12' -ForegroundColor Red; exit 1 }
$pyArgs = @($py | Select-Object -Skip 1)
if (-not (Test-Path .venv\Scripts\python.exe)) { & $py[0] @pyArgs -m venv .venv }
Write-Host "PulseIQ8_AI on $(& .venv\Scripts\python.exe --version)"
& .venv\Scripts\python.exe -m pip install -q -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host 'pip install failed' -ForegroundColor Red; exit 1 }

# load .env into this process (KEY=VALUE; quotes removed; comments and blank lines skipped)
foreach ($line in Get-Content .env) {
    if ($line -match '^\s*#' -or $line -notmatch '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') { continue }
    $name = $Matches[1]; $value = $Matches[2].Trim()
    if ($value -match '^"(.*)"$' -or $value -match "^'(.*)'$") { $value = $Matches[1] }
    elseif ($value -match '^(.*?)\s+#') { $value = $Matches[1] }
    Set-Item -Path "Env:$name" -Value $value
}
# Office proxies must not be used for Ollama / AnalyticsIQ on this machine
$env:NO_PROXY = (@($env:NO_PROXY, 'localhost', '127.0.0.1', '::1') | Where-Object { $_ }) -join ','
$port = if ($env:PORT) { $env:PORT } else { '8091' }
$Host.UI.RawUI.WindowTitle = "PulseIQ8_AI (port $port)"
& .venv\Scripts\python.exe -m uvicorn app.main:app --http h11 --host 127.0.0.1 --port $port
