# PyTorch Lightning Learning Tutorial

Author: Gurpreet Singh

Date: 5/8/2025

This repo has some fundamental code I used to learn about PyTorch Lightning.

At a high level overview PyTorch Lightning provides a development speedup for research by:
- No longer a concern to correctly configure model to `model.train()` or `model.eval()` within your workflow
- Automatic GPU support, no need to manually configure `device` variable and push the model to said `device`. 
- Easily configure and scale GPU and TPU support
- No longer have to configure:
    - `optimizer.zero_grad()`
    - `loss.backward()`
    - `optimizer.step()`
- No longer have to configure `with torch.no_grad()` or `.detach()` to disable to detach from the PyTorch computation graph.
- Tensorboard support

