# 🐍 Python Environment Setup Guide

## 📍 Dein aktuelles Setup (Stand: 13. August 2025)

### ✅ Was bereits gut läuft:
- **Python 3.12.10** als Basis (stabil und gut unterstützt)
- **Isoliertes Virtual Environment** für das Projekt
- **Saubere Paket-Trennung** (keine globalen Installationen)

### 🎯 Deine Python-Installationen:
```
1. Python 3.12 (Microsoft Store) ← Aktuelle Basis
2. Python 3.13 (Microsoft Store) ← Backup/Zukunft
```

## 🔧 Empfohlene Optimierungen

### 1. Virtual Environment Management

**Aktuelles venv beibehalten:**
```bash
# Aktivieren (wenn nicht aktiv)
.venv\Scripts\Activate.ps1

# Status prüfen
python --version
pip list | measure-object
```

**Zukünftige Projekte:**
```bash
# Neues Projekt starten
cd "C:\Users\fwmoe\Projekte\NeuesProjekt"
python3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Paket-Management verbessern

**Requirements.txt nutzen:**
```bash
# Aktuelle Pakete sichern
pip freeze > requirements.txt

# In neuem Environment installieren
pip install -r requirements.txt
```

**Development vs. Production trennen:**
```bash
# requirements.txt (Production)
streamlit==1.38.0
pandas==2.2.3
plotly==6.2.0

# requirements-dev.txt (Development)
-r requirements.txt
jupyter==1.0.0
black==25.1.0
pytest==8.3.2
```

### 3. Python-Version Management

**Versionswahl für neue Projekte:**
- **Python 3.12**: Für Produktiv-Code (stabil)
- **Python 3.13**: Für Experimente (neueste Features)

**Explizite Versionswahl:**
```bash
# Spezifische Version nutzen
py -3.12 -m venv .venv
py -3.13 -m venv .venv-experimental
```

## 🚀 Quick Setup für neue Projekte

### Template für neues ML-Projekt:

```bash
# 1. Projekt-Ordner erstellen
mkdir "C:\Users\fwmoe\Projekte\NeuesProjekt"
cd "C:\Users\fwmoe\Projekte\NeuesProjekt"

# 2. Virtual Environment
python3.12 -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Basis-Pakete installieren
pip install --upgrade pip
pip install streamlit pandas numpy plotly scikit-learn

# 4. Requirements sichern
pip freeze > requirements.txt

# 5. Git initialisieren
git init
echo ".venv/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "__pycache__/" >> .gitignore
```

## 🔍 Troubleshooting

### Problem: Python nicht gefunden
```bash
# Lösung: Spezifische Version nutzen
py -3.12 --version
# oder
python3.12 --version
```

### Problem: Paket-Konflikte
```bash
# Lösung: Environment neu erstellen
deactivate
rm -rf .venv
python3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Problem: Veraltete Pakete
```bash
# Lösung: Selective Updates
pip list --outdated
pip install --upgrade streamlit pandas plotly
```

## 📋 Wartungs-Checkliste

### Monatlich:
- [ ] `pip list --outdated` ausführen
- [ ] Kritische Updates installieren
- [ ] `requirements.txt` aktualisieren

### Bei neuen Projekten:
- [ ] Spezifische Python-Version wählen
- [ ] Virtual Environment erstellen
- [ ] `.gitignore` mit venv-Ausschluss
- [ ] `requirements.txt` von Anfang an pflegen

### Bei Problemen:
- [ ] `pip freeze > backup_requirements.txt`
- [ ] Environment neu erstellen
- [ ] Pakete aus Backup installieren

## 🎯 Best Practices für dich

1. **Ein venv pro Projekt** - Niemals in globaler Installation arbeiten
2. **Requirements.txt pflegen** - Immer aktuell halten
3. **Python 3.12 als Standard** - Für neue Projekte
4. **Regelmäßige Updates** - Aber vorsichtig testen
5. **Backup vor größeren Änderungen** - pip freeze als Sicherheit

---
*Erstellt: 13. August 2025*
*Für: Lukas Next-Gen PP5 Projekt*
