# LOGIN — Face Recognition System

## Complete Documentation

> **Version:** 0.1.0  
> **Python:** ≥ 3.13  
> **Platform:** Windows (DirectShow camera), adaptable to Linux (V4L2)  
> **Package Manager:** [uv](https://docs.astral.sh/uv/)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Installation & Setup](#3-installation--setup)
4. [CLI Commands Reference](#4-cli-commands-reference)
5. [Project Structure](#5-project-structure)
6. [Configuration Reference](#6-configuration-reference)
7. [Module Documentation](#7-module-documentation)
8. [Pipeline Workflows](#8-pipeline-workflows)
9. [Anti-Spoofing System](#9-anti-spoofing-system)
10. [Debug System](#10-debug-system)
11. [Data Storage](#11-data-storage)
12. [Models Reference](#12-models-reference)
13. [Error Handling](#13-error-handling)
14. [Performance Notes](#14-performance-notes)


## Quick Reference

```bash
# First-time setup
uv sync

# Enroll a face
uv run main.py enroll --name YourName

# Recognize
uv run main.py recognize

# List enrolled faces
uv run main.py list

# Delete a face
uv run main.py delete --name YourName

# Audit for duplicates
uv run main.py audit

# Enhance faces (capture + SR)
uv run main.py enhance

# Debug mode (any command)
uv run main.py --debug recognize
uv run main.py -v --debug enroll --name Test
```
