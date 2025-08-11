#!/usr/bin/env bash
mkdir -p ~/.streamlit/
cat <<- EOF > ~/.streamlit/credentials.toml
[general]
email = "user@example.com"
EOF
cat <<- EOF > ~/.streamlit/config.toml
[server]
headless = true
enableCORS = false
enableXsrfProtection=false
EOF