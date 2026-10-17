#!/bin/bash
source /etc/network_turbo
export OLLAMA_MODELS=/root/autodl-tmp/ollama-models
exec ollama serve
