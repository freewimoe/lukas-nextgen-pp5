# Einfacher Python Helper
function pystatus {
    Write-Host "Python Environment Status" -ForegroundColor Green
    if ($env:VIRTUAL_ENV) {
        Write-Host "Virtual Environment aktiv: $env:VIRTUAL_ENV" -ForegroundColor Green
        python --version
        pip list | Select-Object -First 5
    } else {
        Write-Host "Kein Virtual Environment aktiv" -ForegroundColor Yellow
    }
}

function pyupdate {
    if (-not $env:VIRTUAL_ENV) {
        Write-Host "Kein Virtual Environment aktiv!" -ForegroundColor Red
        return
    }
    pip freeze > "backup_$(Get-Date -Format 'yyyyMMdd').txt"
    pip list --outdated
    pip install --upgrade pip
}

function pyclean {
    if (-not $env:VIRTUAL_ENV) {
        Write-Host "Kein Virtual Environment aktiv!" -ForegroundColor Red
        return
    }
    pip cache purge
    Get-ChildItem -Recurse -Name "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Environment bereinigt!" -ForegroundColor Green
}

Write-Host "Python Helper geladen! Verfügbare Befehle: pystatus, pyupdate, pyclean" -ForegroundColor Green
