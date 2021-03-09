---
name: Bug/Performance issue
about: Use this template for reporting a bug or a performance issue.

---

**Dubugging advice**
[delete this section if it doesn't solve your issue]
- Add a `--opset` flag with the highest possible opset you can use. Some ops only convert in higher opsets.
- Try installing the latest tf2onnx from master. Some bug fixes might not have been released to PyPI. Run `pip uninstall tf2onnx` and `pip install git+https://github.com/onnx/tensorflow-onnx`
- If using a saved model, use the Tensorflow `saved_model_cli` to determine the correct `--tag` and `--signature_def` flags to use. If the signature you need is not listed, use the `--concrete_function` flag to index into the model's defined functions.
- If your model was made in tf1.x, try running tf2onnx in a venv with tensorflow 1.x installed. tf2.x should be able to read tf1 models, but sometimes there are bugs.

**Describe the bug**
A clear and concise description of what the bug is.

**Urgency**
If there are particular important use cases blocked by this or strict project-related timelines, please share more information and dates. If there are no hard deadlines, please specify none.

**System information**
- OS Platform and Distribution (e.g., Linux Ubuntu 16.04):
- Tensorflow Version:
- Python version:

**To Reproduce**
Describe steps/code to reproduce the behavior. Please upload/link the model you are trying to convert if possible.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Additional context**
Add any other context about the problem here. If the issue is about a particular model, please share the model details as well to facilitate debugging.
