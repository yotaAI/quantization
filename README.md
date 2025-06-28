
# Quantization
In this repo you can find the different type of quantization approach where I have prepared my custom simulator for `quantized modules`.

### ✅ When Simulation Is Useful
- Quick prototype or visual debugging
- Training with quant noise simulation (QAT-style)
- Understanding rounding/clipping effects
**But if your goal is:**
- Real speedup
- Mobile or server deployment
- Comparing with papers (like "A White Paper on Quantization")

👉 then you must test the final quantized model using real quantized layers (via `torch.quantization.convert`).


**Note :** simulated quantization might give only `~0.2–1%` higher accuracy than the actual `int8` backend.

### `8-bit Signed Symmetric Quantization`
- Applied on *`Resnet-50`*
- Quantized only `Linear` and `Conv2D` layers.

---
### Links and Paths
- FP32 Resnet Model Validation 👉  [link](./imagenet/main.py)
- Quantized Resnet Validation 👉  [link](./quantization/main.py)

### Scoring

| Model |  Type     | Top-1                | Top-5 |
| :-------- | :------- | :------------------------- |:------------------------- |
| Resnet-50 | FP32 | 76.16 |92.88 |
| Resnet-50 | Int-8 Quant [ Linear, Conv2D ] | 75.56 |92.81 |


---
### Reference

👉 [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295)

👉 [SYQ: Learning Symmetric Quantization For Efficient Deep Neural Networks](https://arxiv.org/abs/1807.00301)