#tail.ps1
param([string]$Agent)
Get-Content -Path "logs\$Agent.log" -Wait -Tail 20