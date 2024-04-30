FROM ubuntu:latest

WORKDIR /workspace

RUN apt update && \
    apt install -y curl build-essential && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# install rye
ENV RYE_HOME="$HOME/.rye"
ENV PATH="$RYE_HOME/shims:$PATH"
RUN curl -sSf https://rye-up.com/get | RYE_NO_AUTO_INSTALL=1 RYE_INSTALL_OPTION="--yes" bash

# activate uv
RUN rye config --set-bool behavior.use-uv=true
COPY pyproject.toml requirements.lock requirements-dev.lock .python-version README.md ./
