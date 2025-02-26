# Contributing to Granite TimeSeries Forecasting Tool

Thank you for your interest in contributing to the Granite TimeSeries Forecasting Tool! We welcome your suggestions, bug reports, documentation improvements, and pull requests.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating in this project, you agree to abide by its terms.

## How Can I Contribute?

There are several ways you can contribute to the project:

### Reporting Bugs

- **Open an Issue:** If you encounter any bugs or unexpected behavior, please open a new issue on GitHub. Be sure to include detailed information such as:
  - Steps to reproduce the bug
  - Expected behavior
  - Screenshots or logs (if applicable)
  - Your environment and system details

### Suggesting Enhancements

- **Feature Requests:** If you have ideas for improvements or new features, please submit a feature request on GitHub. Provide as much detail as possible, including:
  - A clear and descriptive title
  - A detailed description of the feature
  - Any examples or use cases

### Pull Requests

Pull requests are the preferred way to contribute code. When you are ready to submit a pull request, please follow these steps:

1. **Fork the Repository:** Click the "Fork" button at the top right of the GitHub repository page to create your own copy of the project.
2. **Clone Your Fork:** Clone your fork to your local machine:
   ```bash
   git clone https://github.com/your_username/granite-forecasting-tool.git
   cd granite-forecasting-tool
   ```
3. **Create a Branch:** Create a new branch for your changes. Use a descriptive branch name that summarizes your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make Your Changes:** Implement your changes, making sure to adhere to the project's coding style and guidelines.
5. **Commit Your Changes:** Write clear, concise commit messages that explain the changes. Follow these guidelines:
   Use the imperative mood in the subject line (e.g., "Fix bug with data splitting").
   Include a brief description of the change and why it was made.

   ```bash
   git add .
   git commit -m "Fix bug with data splitting in preprocessing"
   ```

6. **Push to Your Fork:** Push your changes to your fork on GitHub:

   ```bash
   git push origin feature/your-feature-name

   ```

7. **Open a Pull Request:** Navigate to the original repository and click "New Pull Request." Select your branch and provide a detailed description of your changes.

Pull Request Guidelines

- **Descriptive Title and Description:** Clearly explain what your pull request does, why it is needed, and any relevant background information.
- **Small, Focused Changes:** Try to keep your pull requests small and focused on a single issue or feature.
- **Testing:** Ensure that your changes do not break existing functionality. Add tests if applicable.
- **Documentation:** Update or add documentation as needed.
- **Review Process:** Your pull request will be reviewed by maintainers. Be prepared to make changes based on feedback.

Development Setup

To set up your local development environment:

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/granite-forecasting-tool.git
   cd granite-forecasting-tool

   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate

   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Code Style Guidelines

- **Python Style:** Follow PEP 8 guidelines for Python code.
- **Commit Messages:** Use clear and descriptive commit messages.
- **Documentation:** Ensure that your code is well-documented, including docstrings and inline comments where necessary.

Questions?

If you have any questions, feel free to open an issue or contact one of the maintainers.

Thank you for helping improve the Granite TimeSeries Forecasting Tool!
