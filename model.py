"""
Build Your Own teenygrad: A Tiny Tensor Autograd Engine

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - prod
def prod(shape):
    # Multiply together the elements of a shape tuple to get the total number of elements.
    result = 1
    for dim in shape:
        result *= dim
    return result

# Step 2 - argsort
def argsort(values):
    # Return the indices that would sort values in ascending order.
    # Python's sorted() is stable, preserving the original order of equal elements.
    return sorted(range(len(values)), key=lambda i: values[i])

# Step 3 - make_op_enums
from enum import Enum, auto

def make_op_enums():
    # Create four enum classes naming every supported operation kind.
    class UnaryOps(Enum):
        NEG = auto()
        RELU = auto()
        LOG = auto()
        EXP = auto()
        SQRT = auto()
        SIGMOID = auto()

    class BinaryOps(Enum):
        ADD = auto()
        SUB = auto()
        MUL = auto()
        DIV = auto()
        CMPLT = auto()
        MAX = auto()

    class ReduceOps(Enum):
        SUM = auto()
        MAX = auto()

    class MovementOps(Enum):
        RESHAPE = auto()
        EXPAND = auto()
        PERMUTE = auto()

    return UnaryOps, BinaryOps, ReduceOps, MovementOps

# Step 4 - LazyBuffer
import numpy as np

class LazyBuffer:
    def __init__(self, np_array):
        # Wrap the input as a NumPy array and expose its shape and dtype.
        self._np = np.asarray(np_array)
        self.shape = self._np.shape
        self.dtype = self._np.dtype

# Step 5 - lazybuffer_const
def const(value, shape):
    # Create a new LazyBuffer of the given shape filled with a constant value.
    return LazyBuffer(np.full(shape, value, dtype=np.float32))


LazyBuffer.const = staticmethod(const)

# Step 6 - rand
def rand(shape, seed=None):
    # Return a LazyBuffer of uniform random floats in [0, 1) with the given shape.
    rng = np.random.default_rng(seed)
    return LazyBuffer(rng.random(shape).astype(np.float32))

# Step 7 - lazybuffer_unary_e
def e(self, op):
    # Apply a unary elementwise op (NEG, RELU, LOG, EXP, SQRT, SIGMOID).
    if op.name == "NEG":
        result = -self._np
    elif op.name == "RELU":
        result = np.maximum(self._np, 0)
    elif op.name == "LOG":
        result = np.log(self._np)
    elif op.name == "EXP":
        result = np.exp(self._np)
    elif op.name == "SQRT":
        result = np.sqrt(self._np)
    elif op.name == "SIGMOID":
        result = 1 / (1 + np.exp(-self._np))
    else:
        raise ValueError(f"Unsupported unary operation: {op}")

    return LazyBuffer(result)


LazyBuffer.e = e

# Step 8 - lazybuffer_binary_e
def lazybuffer_binary_e(self, op, other):
    # Apply a binary elementwise op between two LazyBuffers.
    # The inputs are never mutated.
    a = self._np
    b = other._np

    if op.name == "ADD":
        result = a + b
    elif op.name == "SUB":
        result = a - b
    elif op.name == "MUL":
        result = a * b
    elif op.name == "DIV":
        result = a / b
    elif op.name == "CMPLT":
        result = (a < b).astype(np.float32)
    elif op.name == "MAX":
        result = np.maximum(a, b)
    else:
        raise ValueError(f"Unsupported binary operation: {op}")

    return LazyBuffer(result)

# Step 9 - lazybuffer_r
def r(self, op, axis):
    # Reduce the underlying array along the given axis,
    # keeping the reduced dimensions as size 1.
    if op.name == "SUM":
        result = np.sum(self._np, axis=axis, keepdims=True)
    elif op.name == "MAX":
        result = np.max(self._np, axis=axis, keepdims=True)
    else:
        raise ValueError(f"Unsupported reduce operation: {op}")

    return LazyBuffer(result)

# Step 10 - lazybuffer_reshape
def reshape(self, new_shape):
    # Return a new LazyBuffer with the array reshaped to new_shape.
    return LazyBuffer(self._np.reshape(new_shape))

# Step 11 - lazybuffer_expand
def expand(self, new_shape):
    # Broadcast size-1 dimensions to the requested shape.
    # Return a new LazyBuffer without mutating the original buffer.
    result = np.broadcast_to(self._np, new_shape).copy()
    return LazyBuffer(result)

# Step 12 - lazybuffer_permute
def permute(self, order):
    # Return a new LazyBuffer with axes reordered according to order.
    return LazyBuffer(self._np.transpose(order))

# Step 13 - Function
class Function:
    def __init__(self, *tensors):
        # Record whether each input tensor requires gradients.
        self.needs_input_grad = [tensor.requires_grad for tensor in tensors]

        # Preserve None distinctly from False:
        # True if any input requires grad,
        # None if no input is True but at least one is None,
        # otherwise False.
        if any(need is True for need in self.needs_input_grad):
            self.requires_grad = True
        elif any(need is None for need in self.needs_input_grad):
            self.requires_grad = None
        else:
            self.requires_grad = False

        # Only keep parent references when gradients will flow.
        if self.requires_grad:
            self.parents = tuple(
                tensor
                for tensor, need in zip(tensors, self.needs_input_grad)
                if need is True
            )

# Step 14 - function_forward_backward_stubs
def function_forward_backward_stubs():
    # Attach base forward and backward methods to Function.
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    Function.forward = forward
    Function.backward = backward

# Step 15 - apply
@classmethod
def apply(cls, *tensors, **kwargs):
    # Build the Function context.
    ctx = cls(*tensors)

    # Run forward using the input tensors' underlying lazy buffers.
    result = ctx.forward(*(t.lazydata for t in tensors), **kwargs)

    # Wrap the result in a Tensor with the correct gradient requirement.
    out = Tensor(result, requires_grad=ctx.requires_grad)

    # Link the output to the Function when gradients are needed.
    if ctx.requires_grad:
        out._ctx = ctx

    return out


# Provided: attaches apply onto the Function base class. Leave this as-is.
for _obj in list(globals().values()):
    if isinstance(_obj, type):
        for _k in _obj.__mro__:
            if _k.__name__ == 'Function':
                _k.apply = apply

# Step 16 - Neg
class Neg(Function):
    def forward(self, x):
        # Return a LazyBuffer holding the elementwise negation of x.
        UnaryOps, _, _, _ = make_op_enums()
        return x.e(UnaryOps.NEG)

    def backward(self, grad_output):
        # Return the negated incoming gradient.
        UnaryOps, _, _, _ = make_op_enums()
        return grad_output.e(UnaryOps.NEG)

# Step 17 - Relu
class Relu(Function):
    def forward(self, x):
        # Apply ReLU and cache the result for the backward pass.
        UnaryOps, _, _, _ = make_op_enums()
        self.ret = x.e(UnaryOps.RELU)
        return self.ret

    def backward(self, grad_output):
        # Pass the gradient only where the ReLU output is positive.
        _, BinaryOps, _, _ = make_op_enums()
        zero = LazyBuffer.const(0.0, self.ret.shape)
        mask = lazybuffer_binary_e(zero, BinaryOps.CMPLT, self.ret)
        return lazybuffer_binary_e(grad_output, BinaryOps.MUL, mask)

# Step 18 - Log
class Log(Function):
    def forward(self, x):
        # Return the natural log of x and save x for backward.
        UnaryOps, _, _, _ = make_op_enums()
        self.x = x
        return x.e(UnaryOps.LOG)

    def backward(self, grad_output):
        # d/dx log(x) = 1/x, so return grad_output / x.
        _, BinaryOps, _, _ = make_op_enums()
        one = LazyBuffer.const(1.0, self.x.shape)
        reciprocal = lazybuffer_binary_e(one, BinaryOps.DIV, self.x)
        return lazybuffer_binary_e(grad_output, BinaryOps.MUL, reciprocal)

# Step 19 - Exp
class Exp(Function):
    def forward(self, x):
        # Compute the elementwise exponential and cache the output.
        UnaryOps, _, _, _ = make_op_enums()
        self.ret = x.e(UnaryOps.EXP)
        return self.ret

    def backward(self, grad_output):
        # d/dx exp(x) = exp(x), so multiply by the cached output.
        _, BinaryOps, _, _ = make_op_enums()
        return lazybuffer_binary_e(grad_output, BinaryOps.MUL, self.ret)

# Step 20 - Sqrt
class Sqrt(Function):
    def forward(self, x):
        # Compute the elementwise square root and cache it for backward.
        UnaryOps, _, _, _ = make_op_enums()
        self.ret = x.e(UnaryOps.SQRT)
        return self.ret

    def backward(self, grad_output):
        # d/dx sqrt(x) = 1 / (2 * sqrt(x)).
        _, BinaryOps, _, _ = make_op_enums()
        two = LazyBuffer.const(2.0, self.ret.shape)
        denominator = lazybuffer_binary_e(two, BinaryOps.MUL, self.ret)
        reciprocal = lazybuffer_binary_e(
            LazyBuffer.const(1.0, self.ret.shape),
            BinaryOps.DIV,
            denominator,
        )
        return lazybuffer_binary_e(grad_output, BinaryOps.MUL, reciprocal)

# Step 21 - Sigmoid
class Sigmoid(Function):
    def forward(self, x):
        # Return the elementwise logistic activation and cache it for backward.
        UnaryOps, _, _, _ = make_op_enums()
        self.ret = x.e(UnaryOps.SIGMOID)
        return self.ret

    def backward(self, grad_output):
        # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)).
        _, BinaryOps, _, _ = make_op_enums()

        one = LazyBuffer.const(1.0, self.ret.shape)
        one_minus_ret = lazybuffer_binary_e(one, BinaryOps.SUB, self.ret)
        derivative = lazybuffer_binary_e(
            self.ret,
            BinaryOps.MUL,
            one_minus_ret,
        )

        return lazybuffer_binary_e(
            grad_output,
            BinaryOps.MUL,
            derivative,
        )

# Step 22 - Add
class Add(Function):
    def forward(self, x, y):
        # Return the elementwise sum of LazyBuffers x and y.
        _, BinaryOps, _, _ = make_op_enums()
        return lazybuffer_binary_e(x, BinaryOps.ADD, y)

    def backward(self, grad_output):
        # The derivative of x + y with respect to both inputs is 1.
        gx = grad_output if self.needs_input_grad[0] else None
        gy = grad_output if self.needs_input_grad[1] else None
        return gx, gy

# Step 23 - Sub
class Sub(Function):
    def forward(self, x, y):
        # Return the elementwise difference x - y as a LazyBuffer.
        _, BinaryOps, _, _ = make_op_enums()
        return lazybuffer_binary_e(x, BinaryOps.SUB, y)

    def backward(self, grad_output):
        # d(x - y)/dx = 1
        # d(x - y)/dy = -1
        UnaryOps, _, _, _ = make_op_enums()

        gx = grad_output if self.needs_input_grad[0] else None
        gy = grad_output.e(UnaryOps.NEG) if self.needs_input_grad[1] else None

        return gx, gy

# Step 24 - Mul
class Mul(Function):
    def forward(self, x, y):
        # Compute the elementwise product and save inputs for backward.
        _, BinaryOps, _, _ = make_op_enums()
        self.x = x
        self.y = y
        return lazybuffer_binary_e(x, BinaryOps.MUL, y)

    def backward(self, grad_output):
        # d(x * y)/dx = y
        # d(x * y)/dy = x
        _, BinaryOps, _, _ = make_op_enums()

        gx = (
            lazybuffer_binary_e(grad_output, BinaryOps.MUL, self.y)
            if self.needs_input_grad[0]
            else None
        )

        gy = (
            lazybuffer_binary_e(grad_output, BinaryOps.MUL, self.x)
            if self.needs_input_grad[1]
            else None
        )

        return gx, gy

# Step 25 - Div
class Div(Function):
    def forward(self, x, y):
        # Divide x by y and cache the inputs for backward.
        _, BinaryOps, _, _ = make_op_enums()
        self.x = x
        self.y = y
        return lazybuffer_binary_e(x, BinaryOps.DIV, y)

    def backward(self, grad_output):
        # d(x / y)/dx = 1 / y
        # d(x / y)/dy = -x / y^2

        _, BinaryOps, _, _ = make_op_enums()

        gx = None
        gy = None

        if self.needs_input_grad[0]:
            one = LazyBuffer.const(1.0, self.y.shape)
            inv_y = lazybuffer_binary_e(one, BinaryOps.DIV, self.y)
            gx = lazybuffer_binary_e(grad_output, BinaryOps.MUL, inv_y)

        if self.needs_input_grad[1]:
            y_squared = lazybuffer_binary_e(self.y, BinaryOps.MUL, self.y)
            numerator = lazybuffer_binary_e(self.x, BinaryOps.DIV, y_squared)
            neg_numerator = LazyBuffer(
                -numerator._np
            )
            gy = lazybuffer_binary_e(grad_output, BinaryOps.MUL, neg_numerator)

        return gx, gy

# Step 26 - sum_function_forward
class Sum(Function):
    def forward(self, x, axis):
        # Reduce x with ReduceOps.SUM over axis while keeping the reduced dimension.
        _, _, ReduceOps, _ = make_op_enums()

        self.input_shape = x.shape
        self.axis = axis

        return r(x, ReduceOps.SUM, axis)

# Step 27 - sum_function_backward
def backward(self, grad_output):
    # Broadcast the summed gradient back to the original input shape.
    return expand(grad_output, self.input_shape)

# Step 28 - max_function_forward
class Max(Function):
    def forward(self, x, axis):
        # Reduce x with the MAX reduce op along axis and cache values for backward.
        _, _, ReduceOps, _ = make_op_enums()

        self.x = x
        self.axis = axis
        self.ret = r(x, ReduceOps.MAX, axis)

        return self.ret

# Step 29 - max_function_backward
def backward(self, grad_output):
    # Route the gradient only to maximum elements.
    # When multiple elements tie for the maximum, split the gradient evenly.
    _, BinaryOps, ReduceOps, _ = make_op_enums()

    # mask = 1 where x == max, otherwise 0.
    less_than_max = lazybuffer_binary_e(self.x, BinaryOps.CMPLT, self.ret)
    one = LazyBuffer.const(1.0, self.x.shape)
    mask = lazybuffer_binary_e(one, BinaryOps.SUB, less_than_max)

    # Count how many maximum elements occur in each reduced bucket.
    count = r(mask, ReduceOps.SUM, self.axis)

    # Expand the upstream gradient and tie count back to the input shape.
    grad = expand(grad_output, self.x.shape)
    count = expand(count, self.x.shape)

    # Split the gradient evenly among all tied maximum elements.
    grad_per_max = lazybuffer_binary_e(grad, BinaryOps.DIV, count)

    return lazybuffer_binary_e(grad_per_max, BinaryOps.MUL, mask)

Max.backward = backward

# Step 30 - Reshape
class Reshape(Function):
    def forward(self, x, shape):
        # Cache the original input shape and reshape the buffer.
        self.input_shape = x.shape
        return reshape(x, shape)

    def backward(self, grad_output):
        # Reshape the incoming gradient back to the original input shape.
        return reshape(grad_output, self.input_shape)

# Step 31 - expand_function_forward (not yet solved)
# TODO: implement

# Step 32 - expand_function_backward (not yet solved)
# TODO: implement

# Step 33 - permute_function_forward_backward (not yet solved)
# TODO: implement

# Step 34 - Tensor (not yet solved)
# TODO: implement

# Step 35 - tensor_from_data (not yet solved)
# TODO: implement

# Step 36 - tensor_creation_helpers (not yet solved)
# TODO: implement

# Step 37 - tensor_randn (not yet solved)
# TODO: implement

# Step 38 - build_topological_order (not yet solved)
# TODO: implement

# Step 39 - tensor_backward (not yet solved)
# TODO: implement

# Step 40 - bind_unary_tensor_methods (not yet solved)
# TODO: implement

# Step 41 - broadcasted (not yet solved)
# TODO: implement

# Step 42 - bind_binary_tensor_methods (not yet solved)
# TODO: implement

# Step 43 - bind_movement_tensor_methods (not yet solved)
# TODO: implement

# Step 44 - bind_reduce_tensor_methods (not yet solved)
# TODO: implement

# Step 45 - tensor_mean (not yet solved)
# TODO: implement

# Step 46 - tensor_transpose (not yet solved)
# TODO: implement

# Step 47 - tensor_matmul_2d (not yet solved)
# TODO: implement

# Step 48 - tensor_softmax (not yet solved)
# TODO: implement

# Step 49 - tensor_log_softmax (not yet solved)
# TODO: implement

# Step 50 - sparse_categorical_cross_entropy (not yet solved)
# TODO: implement

# Step 51 - Linear (not yet solved)
# TODO: implement

# Step 52 - MLP (not yet solved)
# TODO: implement

# Step 53 - sgd_step (not yet solved)
# TODO: implement

# Step 54 - zero_grad (not yet solved)
# TODO: implement

# Step 55 - make_toy_digit_dataset (not yet solved)
# TODO: implement

# Step 56 - accuracy (not yet solved)
# TODO: implement

# Step 57 - train_mlp (not yet solved)
# TODO: implement

# Step 58 - evaluate_mlp (not yet solved)
# TODO: implement

