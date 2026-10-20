"""Scripts executed inside the pinned role environments (quantize/compile/verify).

They import heavy packages (torch, Olive, ONNX Runtime GenAI) that are not installed in the
CLI's own environment, so the CLI never imports these modules; it passes their file paths
to the role environment's Python.
"""
