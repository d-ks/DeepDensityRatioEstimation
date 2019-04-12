# DeepLogReg

![estimation_animation_LogReg](./LogReg.gif "Estimating density ratio by DeepLogReg")

*Estimating animation (true: dotted line, estimated:red line)*

---

Chainer Implementation of DeepLogReg, proposed in following paper.
* **Eiji Uchibe** (2018) - *Model-Free Deep Inverse Reinforcement Learning by Logistic Regression.* - [link](https://link.springer.com/article/10.1007/s11063-017-9702-7)

## Requirements
* Python3
* numpy
* Chainer

## Usage
First, run gen.py to generate dataset. This generates 10000 samples according to two Normal distributions.
```
python3 gen.py
```
Next, run LogReg.py, 

```
python3 LogReg.py
```

and you will get estimated density ratio in "r_LogReg.csv".

## License

This project is licensed under the MIT License.
