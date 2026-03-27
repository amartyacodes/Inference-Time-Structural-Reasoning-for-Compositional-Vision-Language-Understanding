# Contributing to LLaVA Evaluation Framework

Thank you for your interest in contributing! Here's how you can help:

## Code Style

- Follow PEP 8 conventions
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep lines under 100 characters

## Setup for Development

```bash
# Clone and setup
git clone <repo>
cd llava_eval
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install development tools (optional)
pip install black flake8 mypy
```

## Making Changes

1. **Fork and create a branch**
   ```bash
   git checkout -b fix/issue-name
   ```

2. **Make your changes** with clear commit messages

3. **Test your code**
   - Ensure it runs without errors
   - Test with `--max_samples 10` for quick validation

4. **Submit a Pull Request** with:
   - Clear description of changes
   - Reference to related issues
   - Test results (if applicable)

## Adding New Features

- New evaluation methods → add to `final.py` or create new script
- New ablation types → extend `qwen_3_ablation.py`
- New datasets → create separate evaluation script
- Bug fixes → create issue first, reference in PR

## Reporting Issues

Include:
- Python version and environment setup
- Full error message and traceback
- Reproduction steps
- GPU/hardware info
- Any relevant model/dataset information

## Questions?

Open a discussion or issue on GitHub!
