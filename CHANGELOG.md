# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-03-26

### Added
- Unified multi-model evaluation framework
- Support for CLIP, BLIP, LLaVA, Qwen3-VL models
- Scene graph integration for compositional reasoning
- Comprehensive ablation studies (caption-level masking, swapping, shuffling)
- Attention probing and patching analysis
- Multi-turn dialogue generation
- Full documentation and development guides
- GitHub Actions CI/CD pipeline
- Black/flake8 linting configuration

### Fixed
- Memory management for large model inference
- GPU out-of-memory error handling

### Changed
- Reorganized codebase for clarity and maintainability

---

## Future Versions

### Planned for v1.1.0
- [ ] Support for additional models (e.g., GPT-4V)
- [ ] Distributed evaluation across multiple GPUs
- [ ] Web dashboard for result visualization
- [ ] Caching mechanism for repeated evaluations

### Planned for v1.2.0
- [ ] Additional ablation types
- [ ] Fine-tuning capabilities
- [ ] Export to multiple formats (CSV, PDF)
