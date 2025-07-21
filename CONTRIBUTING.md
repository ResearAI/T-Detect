# Contributing to T-Detect

We welcome contributions to T-Detect! This document provides guidelines for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/t-detect.git
   cd t-detect
   ```
3. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Or install with development extras
pip install -e ".[dev]"
```

### Environment Setup

Set up your environment variables:

```bash
export DEVICE=cuda  # or cpu
export CACHE_DIR=./.cache
export HF_HOME=./.cache/huggingface
```

## Code Style

We follow standard Python coding conventions:

### Formatting

- Use [Black](https://black.readthedocs.io/) for code formatting:
  ```bash
  black scripts/
  ```

- Use [isort](https://pycqa.github.io/isort/) for import sorting:
  ```bash
  isort scripts/
  ```

### Linting

- Use [flake8](https://flake8.pycqa.org/) for linting:
  ```bash
  flake8 scripts/
  ```

### Documentation

- Write clear docstrings for all functions and classes
- Use type hints where appropriate
- Document complex algorithms with mathematical formulations
- Include examples in docstrings when helpful

Example:
```python
def get_t_discrepancy_analytic(logits_ref: torch.Tensor, 
                              logits_score: torch.Tensor, 
                              labels: torch.Tensor, 
                              nu: float = 5) -> float:
    """
    Compute T-Detect discrepancy using heavy-tailed normalization.
    
    Args:
        logits_ref: Reference model logits [batch_size, seq_len, vocab_size]
        logits_score: Scoring model logits [batch_size, seq_len, vocab_size]
        labels: Ground truth labels [batch_size, seq_len]
        nu: Degrees of freedom for t-distribution
        
    Returns:
        T-Detect score (lower = more likely AI-generated)
        
    Example:
        >>> score = get_t_discrepancy_analytic(ref_logits, score_logits, labels)
        >>> print(f"T-Detect score: {score:.4f}")
    """
```

## Testing

### Running Tests

```bash
# Run core algorithm tests
python test_t_detect.py

# Run simple examples
python simple_example.py

# Run demo (requires models)
python scripts/demo.py --text "Your test text here"
```

### Writing Tests

- Add tests for new functionality
- Test edge cases and error conditions
- Include tests for different parameter settings
- Test robustness to various input types

Example test structure:
```python
def test_t_detect_algorithm():
    """Test T-Detect core functionality."""
    # Create test data
    logits_ref = torch.randn(1, 10, 100)
    logits_score = torch.randn(1, 10, 100)
    labels = torch.randint(0, 100, (1, 10))
    
    # Test basic functionality
    score = get_t_discrepancy_analytic(logits_ref, logits_score, labels)
    assert isinstance(score, float)
    assert not np.isnan(score)
    assert not np.isinf(score)
```

## Submitting Changes

### Pull Request Process

1. Ensure your code follows the style guidelines
2. Add or update tests as needed
3. Update documentation if you're changing functionality
4. Run the test suite to ensure nothing is broken
5. Write a clear commit message describing your changes

### Commit Message Format

Use clear, descriptive commit messages:

```
feat: add support for dynamic threshold adjustment
fix: handle edge case in t-distribution normalization
docs: update README with new installation instructions
test: add robustness tests for adversarial text
```

### Pull Request Description

Include in your PR description:

- What changes you made and why
- Any breaking changes
- How to test the changes
- References to related issues

## Reporting Issues

When reporting issues, please include:

### Bug Reports

- Python version and operating system
- T-Detect version
- Steps to reproduce the issue
- Expected vs actual behavior
- Error messages and stack traces
- Sample code that demonstrates the issue

### Feature Requests

- Clear description of the desired functionality
- Use cases and motivation
- Possible implementation approaches
- Examples of similar features in other tools

### Security Issues

For security-related issues, please email us directly rather than opening a public issue.

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Be patient with newcomers

### Scope

This Code of Conduct applies to all project spaces, including:

- GitHub repositories
- Issue trackers
- Discussions
- Documentation

## Development Guidelines

### Adding New Detectors

If you're adding a new detector:

1. Inherit from `DetectorBase`
2. Implement the `compute_crit` method
3. Add configuration in `scripts/detectors/configs/`
4. Register in `scripts/detectors/__init__.py`
5. Add tests and documentation

Example:
```python
class NewDetector(DetectorBase):
    def __init__(self, config_name):
        super().__init__(config_name)
        # Initialize your detector
        
    def compute_crit(self, text: str) -> float:
        """Compute detection criterion for input text."""
        # Your detection logic here
        return score
```

### Adding New Features

For new features:

1. Discuss the feature in an issue first
2. Keep changes focused and atomic
3. Maintain backward compatibility when possible
4. Add comprehensive tests
5. Update documentation

### Performance Considerations

- Profile code for performance bottlenecks
- Use efficient algorithms and data structures
- Consider memory usage for large datasets
- Optimize GPU usage when applicable

## Getting Help

If you need help:

- Check the [README](README.md) for basic usage
- Look at existing issues for similar problems
- Ask questions in GitHub Discussions
- Review the codebase for examples

## Recognition

Contributors will be acknowledged in:

- README.md contributors section
- Release notes for significant contributions
- Academic papers (with permission)

Thank you for contributing to T-Detect!