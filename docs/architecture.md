# OpenViking OpenVINO Services

This repository standardizes the local OpenVINO service stack for OpenViking on ZSpace NAS.

## Goals

- One shared OpenVINO runtime image.
- Two independent service layers: embedding and intent.
- Production-friendly build, run, and rollback layout.
- No ad-hoc runtime pip installation in containers.

## Current Status

- Shared base image scaffolded.
- Service shell layout being productionized.
- Intent generation loop and OpenViking contract alignment still need verification.
