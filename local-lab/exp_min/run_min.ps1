Set-Location D:\knowleen\local-lab\exp_min
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUNBUFFERED='1'
& D:\knowleen\local-lab\.venv\Scripts\python.exe run_min.py *> D:\knowleen\local-lab\logs\run_min.log
"EXIT $LASTEXITCODE" | Out-File -Append -Encoding utf8 D:\knowleen\local-lab\logs\run_min.log