FROM ubuntu:22.04

# Install basic tools
RUN apt-get update && apt-get install -y \
    curl \
    git \
    vim \
    nano \
    python3 \
    python3-pip \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
WORKDIR /workspace

# Set up a non-root user
RUN useradd -m -s /bin/bash aiuser && \
    chown -R aiuser:aiuser /workspace

USER aiuser

CMD ["/bin/bash"]
