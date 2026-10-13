# Contributing to Whisper Hindi Fine-tuning

Thank you for your interest in contributing to this project! This guide will help you get started.

## 🚀 Quick Start for Contributors

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/yourusername/whisper-hindi-finetuning.git
   cd whisper-hindi-finetuning
   ```
3. **Set up** the development environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## 🎯 Areas for Contribution

### High Priority
- **Performance improvements**: Optimize training speed and memory usage
- **Additional languages**: Extend the approach to other Indic languages
- **Better evaluation**: Add more comprehensive metrics and benchmarks
- **Documentation**: Improve code comments and user guides

### Medium Priority
- **Data augmentation**: Implement audio augmentation techniques
- **Model variants**: Support for different Whisper model sizes
- **Deployment**: Add inference optimization and deployment guides
- **Testing**: Add unit tests and integration tests

### Low Priority
- **UI/UX**: Create a simple web interface for testing
- **Visualization**: Add training progress and result visualization
- **Packaging**: Create pip-installable package

## 📝 Development Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for all functions and classes
- Keep functions focused and small

### Commit Messages
Use clear, descriptive commit messages:
```
feat: add data augmentation for audio files
fix: resolve memory leak in training loop
docs: update installation instructions
test: add unit tests for dataset utils
```

### Pull Request Process
1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes and test them
3. Update documentation if needed
4. Commit your changes with clear messages
5. Push to your fork and create a Pull Request
6. Describe your changes and link any relevant issues

## 🧪 Testing

Before submitting a PR, please ensure:
- [ ] Your code runs without errors
- [ ] Training script completes successfully (if modified)
- [ ] Documentation is updated for new features
- [ ] No large files are accidentally committed

### Running Tests
```bash
# Test training pipeline
python notebook/quick_train.py

# Test evaluation
python notebook/final_evaluation.py

# Test model inference
python notebook/test_model_hindi.py path/to/test/audio.wav
```

## 📋 Reporting Issues

When reporting bugs or requesting features:

1. **Search existing issues** first
2. **Use issue templates** if available
3. **Provide clear reproduction steps** for bugs
4. **Include system information** (OS, Python version, etc.)
5. **Add relevant logs or error messages**

## 🤝 Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain a positive environment

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: For private matters or security issues

## 🏷️ Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes for significant contributions
- Project documentation

Thank you for contributing to open-source ASR research! 🎉