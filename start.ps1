$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $env:AGENT_MODEL_URL -and -not $env:AGENT_MODEL_NAME) {
    try {
        $localModels = Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 3
        if ($localModels.models.name -contains 'gemma3:4b') {
            $env:AGENT_MODEL_URL = 'http://127.0.0.1:11434/api/chat'
            $env:AGENT_MODEL_NAME = 'gemma3:4b'
            Write-Host 'Using existing local gemma3:4b through Ollama.'
        }
    } catch {
        Write-Host 'No local Ollama model detected. Simulation and verified tools remain available.'
    }
}
if (Test-Path -LiteralPath '.venv/Scripts/python.exe') {
    & '.venv/Scripts/python.exe' server.py
} else {
    & python server.py
}
