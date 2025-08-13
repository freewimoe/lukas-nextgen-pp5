# 🚀 Quick Python Environment Manager für dein System
# Speichere diese Datei als: python_manager.ps1

function Show-PythonStatus {
    Write-Host "🐍 Python Environment Status" -ForegroundColor Green
    Write-Host "=============================" -ForegroundColor Green
    
    # Virtual Environment Status
    if ($env:VIRTUAL_ENV) {
        Write-Host "✅ Virtual Environment aktiv: $env:VIRTUAL_ENV" -ForegroundColor Green
        python --version
        Write-Host ""
        Write-Host "📦 Installierte Pakete:" -ForegroundColor Yellow
        pip list | Select-Object -First 10
    } else {
        Write-Host "⚠️  Kein Virtual Environment aktiv" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "🎯 Verfügbare Python-Versionen:" -ForegroundColor Blue
    Get-Command python* -ErrorAction SilentlyContinue | Select-Object Name, Source
}

function New-ProjectSetup {
    param(
        [Parameter(Mandatory=$true)]
        [string]$ProjectName,
        
        [string]$PythonVersion = "3.12"
    )
    
    Write-Host "🚀 Erstelle neues Python-Projekt: $ProjectName" -ForegroundColor Green
    
    # Projekt-Ordner erstellen
    $ProjectPath = "C:\Users\$env:USERNAME\Projekte\$ProjectName"
    New-Item -ItemType Directory -Path $ProjectPath -Force
    Set-Location $ProjectPath
    
    # Virtual Environment erstellen
    Write-Host "📦 Erstelle Virtual Environment..." -ForegroundColor Yellow
    & "python$PythonVersion" -m venv .venv
    
    # Aktivieren
    & ".venv\Scripts\Activate.ps1"
    
    # Basis-Setup
    pip install --upgrade pip
    
    # Basis Requirements
    @"
# Basis Data Science Stack
pandas>=2.2.0
numpy>=1.24.0
plotly>=6.0.0
streamlit>=1.35.0

# ML Stack
scikit-learn>=1.3.0
joblib>=1.3.0

# Development
jupyter>=1.0.0
"@ | Out-File -FilePath "requirements.txt" -Encoding UTF8
    
    # .gitignore erstellen
    @"
# Python
.venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis

# Jupyter Notebook
.ipynb_checkpoints

# VS Code
.vscode/

# OS
.DS_Store
Thumbs.db
"@ | Out-File -FilePath ".gitignore" -Encoding UTF8
    
    Write-Host "✅ Projekt $ProjectName erfolgreich erstellt!" -ForegroundColor Green
    Write-Host "📍 Pfad: $ProjectPath" -ForegroundColor Blue
    Write-Host "💡 Nächste Schritte:" -ForegroundColor Yellow
    Write-Host "   1. pip install -r requirements.txt" -ForegroundColor White
    Write-Host "   2. git init" -ForegroundColor White
    Write-Host "   3. Entwicklung starten!" -ForegroundColor White
}

function Update-ProjectDependencies {
    Write-Host "🔄 Aktualisiere Projekt-Dependencies..." -ForegroundColor Green
    
    if (-not $env:VIRTUAL_ENV) {
        Write-Host "❌ Kein Virtual Environment aktiv! Aktiviere zuerst .venv" -ForegroundColor Red
        return
    }
    
    # Backup erstellen
    pip freeze > "requirements_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
    Write-Host "✅ Backup erstellt" -ForegroundColor Green
    
    # Updates prüfen
    Write-Host "📋 Verfügbare Updates:" -ForegroundColor Yellow
    pip list --outdated
    
    # Sicherheitsupdates
    Write-Host "🔐 Installiere Sicherheitsupdates..." -ForegroundColor Yellow
    pip install --upgrade pip setuptools wheel
    
    # Neue requirements.txt erstellen
    pip freeze > requirements_updated.txt
    Write-Host "✅ Updates abgeschlossen!" -ForegroundColor Green
}

function Clean-PythonEnvironment {
    Write-Host "🧹 Bereinige Python Environment..." -ForegroundColor Green
    
    if (-not $env:VIRTUAL_ENV) {
        Write-Host "❌ Kein Virtual Environment aktiv!" -ForegroundColor Red
        return
    }
    
    # Cache leeren
    pip cache purge
    
    # Temp-Dateien entfernen
    Get-ChildItem -Path . -Recurse -Name "__pycache__" | Remove-Item -Recurse -Force
    Get-ChildItem -Path . -Recurse -Name "*.pyc" | Remove-Item -Force
    
    Write-Host "✅ Environment bereinigt!" -ForegroundColor Green
}

# Aliase für einfache Nutzung
Set-Alias -Name "pystatus" -Value Show-PythonStatus
Set-Alias -Name "pynew" -Value New-ProjectSetup
Set-Alias -Name "pyupdate" -Value Update-ProjectDependencies
Set-Alias -Name "pyclean" -Value Clean-PythonEnvironment

Write-Host "🎯 Python Manager geladen!" -ForegroundColor Green
Write-Host "💡 Verfügbare Befehle:" -ForegroundColor Yellow
Write-Host "   pystatus           - Zeige Python-Status" -ForegroundColor White
Write-Host "   pynew [name]       - Neues Projekt erstellen" -ForegroundColor White
Write-Host "   pyupdate           - Dependencies aktualisieren" -ForegroundColor White
Write-Host "   pyclean            - Environment bereinigen" -ForegroundColor White
