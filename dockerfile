FROM mambaorg/micromamba

USER root

# Install basic dependencies
RUN apt-get update && \
    apt-get install -y \
    rsync

# Set working directory
WORKDIR /app

# Set up environment
COPY environment.yaml pyproject.toml ./
RUN micromamba env create -f environment.yaml
COPY twodimfim/ twodimfim/
RUN micromamba run -n app pip install -e . && \
    micromamba clean --all --yes

# Copy app
COPY app/ app/

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit
CMD ["/opt/conda/envs/app/bin/python", "-m", "streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
