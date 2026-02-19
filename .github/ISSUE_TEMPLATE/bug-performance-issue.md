---
name: Bug/Performance issue
about: Use this template for reporting a bug or a performance issue.
title: ''
labels: 'bug'
assignees: ''
---

<!--
**Dubugging advice**
- Add a `--opset` flag with the highest possible opset you can use. Some ops only convert in higher opsets.
- Try installing the latest tf2onnx from main. Some bug fixes might not have been released to PyPI. Run `pip uninstall tf2onnx` and `pip install git+https://github.com/onnx/tensorflow-onnx`
- If using a saved model, use the Tensorflow `saved_model_cli` to determine the correct `--tag` and `--signature_def` flags to use. If the signature you need is not listed, use the `--concrete_function` flag to index into the model's defined functions.
 -->

**Describe the bug**
<!-- Please describe a clear and concise description of what the bug is. -->

**Urgency**
<!-- If there are particular important use cases blocked by this or strict project-related timelines, please share more information and dates. If there are no hard deadlines, please specify none. -->

**System information**
- OS Platform and Distribution (e.g., Linux Ubuntu 18.04*):
- TensorFlow Version:
- Python version:
- ONNX version (if applicable, e.g. 1.11*):
- ONNXRuntime version (if applicable, e.g. 1.11*):


**To Reproduce**
<!-- Describe steps/code/command to reproduce the behavior. Please upload/link the model you are trying to convert if possible. -->

**Screenshots**
<!-- If applicable, add screenshots to help explain your problem. -->

**Additional context**
<!-- Add any other context about the problem here. If the issue is about a particular model, please share the model details as well to facilitate debugging. -->
