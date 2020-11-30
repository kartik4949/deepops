# DeepOps
A Mini Deep learning library, accelerated on GPUs with PyCuda.

No no no.. I havent wrote this library on tensorflow or torch, it is a standalone machine. :P

Implemented backpropogations using reverse traversal, Support gradients and GPU operations, You can make a mini neural network (FNN) and use the in-house optimizer to train it on a dataset (e.g MNIST).

Tip: always give your tensor a funny name! :)

Note: Only for Educational Usage.

![alt text](https://cdn.pixabay.com/photo/2017/11/14/18/31/mushroom-2949539_960_720.jpg)

# Installation.
```
pip install deepop
```

# Tensor.
```
a = Tensor([1,2,3,4,5])
# deepop tensor 
```
# Attach to a cuda device.
```
a = Tensor([1,2,3,4,5])
a.device("gpu:0") # attach to gpu device.
```
# Check the Device.
```
a.where
# 'cpu'
```
# Addition.
```
a = Tensor([1.0,2.0])
print(a + a)
# GPU Operation
```
# Multiplication.
```
a = Tensor([1.0, 2.0])
print(a.mul(a))
print(a * a)
```
# Calculate Gradients.
```
Tensor = dp.Tensor
a1 = Tensor([1.0, 3.0, 1.0])
b1 = Tensor([7.0, 3.0, 5.0])

a2 = Tensor([4.0, 3.0, 1.0])
a3 = Tensor([3.0, 3.0, 1.0])
a4 = Tensor([7.0, 1.0, 6.0])
b2 = Tensor([1.0, 21.0, 12.0])

c = a1 * b1 + a3
d = a2 * b2 + a4
out = c * d

# backward
out.backward()

print(out.grad)
print(a1.grad)

```
# Run Tests.
```
python -m pytest -s

```

# Contribution is highly appreciated.
Please contribute to my work.

# TODOs
* write more tests...
* need a optimizer.
* support more operations.

# License
MIT

