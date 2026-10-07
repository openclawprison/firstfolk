$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (Test-Path -LiteralPath '.venv\Scripts\python.exe') {
    & '.venv\Scripts\python.exe' world_server.py @args
} else {
    python world_server.py @args
}
