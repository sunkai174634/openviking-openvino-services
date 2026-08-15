# openviking-openvino-services

OpenViking local OpenVINO services for NAS iGPU.

This repository contains a shared OpenVINO 2026.3 base image and two thin service layers:

- `embedding`: OpenAI-compatible `/v1/embeddings`
- `intent`: OpenViking query planner / intent analysis service

The design goal is one shared runtime image, two independent services, and a production-oriented deployment layout.
