# twodimfim
2D hydrodynamic flood modeling to support OWP flood inundation mapping (FIM) efforts

## Overview

This repository contains a Streamlit web application for 2D hydrodynamic flood modeling to support the Office of Water Prediction (OWP) flood inundation mapping efforts.

## Quick Start

### Using Docker Compose (Recommended)

```bash
docker-compose up
```

The application will be available at `http://localhost:8501`

### Using Docker

```bash
docker build -t twodimfim .
docker run -p 8501:8501 twodimfim
```

### Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Development

### Using DevContainer

This repository includes a devcontainer configuration for VS Code. Open the repository in VS Code and select "Reopen in Container" when prompted.

### Using GitHub Container Registry

Pre-built images are automatically pushed to GitHub Container Registry on each commit to main:

```bash
docker pull ghcr.io/dewberry/twodimfim:latest
docker run -p 8501:8501 ghcr.io/dewberry/twodimfim:latest
```

## CI/CD

GitHub Actions automatically builds and pushes Docker images to GitHub Container Registry when:
- Code is pushed to `main` or `develop` branches
- A new tag is created (e.g., `v1.0.0`)
- Pull requests are opened (build only, no push)

## Project Structure

```
.
├── .devcontainer/          # VS Code devcontainer configuration
│   └── devcontainer.json
├── .github/
│   └── workflows/          # GitHub Actions CI/CD workflows
│       └── docker-build-push.yml
├── app.py                  # Streamlit application
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile              # Container image definition
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## License

[Add license information]
