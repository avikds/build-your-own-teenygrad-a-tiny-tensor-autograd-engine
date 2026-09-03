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

Sum.backward = backward

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

# Step 31 - expand_function_forward
def expand_function_forward(ctx, x, shape):
    # Cache the original input shape for the backward pass.
    ctx.input_shape = x.shape

    # Delegate broadcasting to the standalone expand helper.
    return expand(x, shape)

# Step 32 - expand_function_backward
def expand_function_backward(ctx, grad_output):
    # Sum the incoming gradient over the axes that were broadcast.
    input_shape = ctx.input_shape
    output_shape = grad_output.shape

    # Align the input shape to the right side of the output shape.
    ndim_diff = len(output_shape) - len(input_shape)
    aligned_input_shape = (1,) * ndim_diff + tuple(input_shape)

    # Reduce every axis that was introduced or expanded from size 1.
    axes = tuple(
        i
        for i, (in_dim, out_dim) in enumerate(
            zip(aligned_input_shape, output_shape)
        )
        if in_dim == 1 and out_dim != 1
    )

    if axes:
        _, _, ReduceOps, _ = make_op_enums()
        grad_output = r(grad_output, ReduceOps.SUM, axes)

    return reshape(grad_output, input_shape)

# Step 33 - permute_function_forward_backward
def permute_function_forward_backward():
    def forward(ctx, x, order):
        # Store the axis permutation and reorder the buffer.
        ctx.order = order
        return permute(x, order)

    def backward(ctx, grad_output):
        # Apply the inverse permutation to restore the original axis order.
        inverse_order = argsort(ctx.order)
        return permute(grad_output, inverse_order)

    return forward, backward

# Step 34 - Tensor
class Tensor:
    def __init__(self, data, requires_grad=False, _ctx=None):
        # Reuse an existing LazyBuffer or wrap the input data.
        self.lazydata = data if isinstance(data, LazyBuffer) else LazyBuffer(
            np.asarray(data, dtype=np.float32)
        )

        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = _ctx

    @property
    def data(self):
        return self.lazydata

    @data.setter
    def data(self, value):
        self.lazydata = value if isinstance(value, LazyBuffer) else LazyBuffer(
            np.asarray(value, dtype=np.float32)
        )

    @property
    def shape(self):
        return self.lazydata.shape

    @property
    def dtype(self):
        return self.lazydata.dtype

    def numpy(self):
        return self.lazydata._np

# Step 35 - tensor_from_data
def tensor_from_data(data, requires_grad=False):
    # Reuse an existing LazyBuffer directly.
    if isinstance(data, LazyBuffer):
        return Tensor(data, requires_grad=requires_grad)

    # Convert ordinary data to float32 before wrapping it.
    return Tensor(
        LazyBuffer(np.asarray(data, dtype=np.float32)),
        requires_grad=requires_grad,
    )

# Step 36 - tensor_creation_helpers
def tensor_creation_helpers():
    # Return helpers that create constant-filled Tensors.
    def zeros_fn(shape):
        return Tensor(LazyBuffer.const(0.0, shape))

    def ones_fn(shape):
        return Tensor(LazyBuffer.const(1.0, shape))

    def full_fn(shape, value):
        return Tensor(LazyBuffer.const(value, shape))

    return zeros_fn, ones_fn, full_fn

# Step 37 - tensor_randn
def tensor_randn(shape, seed=None, requires_grad=False):
    # Generate reproducible uniform samples.
    uniform = rand(shape, seed=seed)

    # Box-Muller transform: U(0, 1) -> N(0, 1).
    u = uniform._np

    # Avoid log(0) while preserving the intended [0, 1) distribution.
    eps = np.finfo(np.float32).tiny
    u1 = np.maximum(u, eps)
    u2 = np.random.default_rng(seed).random(shape).astype(np.float32)

    gaussian = (
        np.sqrt(-2.0 * np.log(u1))
        * np.cos(2.0 * np.pi * u2)
    ).astype(np.float32)

    return Tensor(LazyBuffer(gaussian), requires_grad=requires_grad)

# Step 38 - build_topological_order
def build_topological_order(tensor):
    # DFS through the computation graph, adding each node after its parents.
    visited = set()
    order = []

    def dfs(node):
        if id(node) in visited:
            return

        visited.add(id(node))

        if node._ctx is not None:
            for parent in getattr(node._ctx, "parents", ()):
                dfs(parent)

        order.append(node)

    dfs(tensor)
    return order

# Step 39 - tensor_backward
def tensor_backward(tensor):
    # Seed the root tensor's gradient with ones.
    tensor.grad = Tensor(
        LazyBuffer.const(1.0, tensor.shape),
        requires_grad=False,
    )

    # Process the graph in reverse topological order.
    topo = build_topological_order(tensor)
    _, BinaryOps, _, _ = make_op_enums()

    for node in reversed(topo):
        if node._ctx is None or node.grad is None:
            continue

        grads = node._ctx.backward(node.grad.data)

        if grads is None:
            continue

        if not isinstance(grads, tuple):
            grads = (grads,)

        parents = node._ctx.parents

        # Handle both possible parent layouts:
        # 1. parents contains every input
        # 2. parents contains only inputs requiring gradients
        if len(parents) == len(grads):
            parent_grads = zip(parents, grads)
        else:
            parent_grads = zip(
                (p for p in parents if p.requires_grad),
                (g for g in grads if g is not None),
            )

        for parent, grad in parent_grads:
            # Never populate .grad for tensors that do not require it.
            if not parent.requires_grad or grad is None:
                continue

            if parent.grad is None:
                parent.grad = Tensor(grad, requires_grad=False)
            else:
                parent.grad.data = lazybuffer_binary_e(
                    parent.grad.data,
                    BinaryOps.ADD,
                    grad,
                )

# Step 40 - bind_unary_tensor_methods
def bind_unary_tensor_methods():
    # Map each unary Tensor operation to its corresponding Function.apply.
    return {
        "neg": lambda x: Neg.apply(x),
        "relu": lambda x: Relu.apply(x),
        "log": lambda x: Log.apply(x),
        "exp": lambda x: Exp.apply(x),
        "sqrt": lambda x: Sqrt.apply(x),
        "sigmoid": lambda x: Sigmoid.apply(x),
    }

# Step 41 - broadcasted
def broadcasted(x, y):
    # Determine the common broadcast shape using NumPy broadcasting rules.
    x_shape = x.shape
    y_shape = y.shape

    max_ndim = max(len(x_shape), len(y_shape))

    x_aligned = (1,) * (max_ndim - len(x_shape)) + tuple(x_shape)
    y_aligned = (1,) * (max_ndim - len(y_shape)) + tuple(y_shape)

    common_shape = []

    for xd, yd in zip(x_aligned, y_aligned):
        if xd == yd:
            common_shape.append(xd)
        elif xd == 1:
            common_shape.append(yd)
        elif yd == 1:
            common_shape.append(xd)
        else:
            raise ValueError(
                f"Shapes {x_shape} and {y_shape} are not broadcastable"
            )

    common_shape = tuple(common_shape)

    # Keep tensors with the common shape unchanged.
    bx = x
    by = y

    # Construct the differentiable Expand Function locally.
    Expand = type(
        "Expand",
        (Function,),
        {
            "forward": expand_function_forward,
            "backward": expand_function_backward,
        },
    )

    # Expand only tensors whose shapes need broadcasting.
    if x.shape != common_shape:
        bx = Expand.apply(x, shape=common_shape)

    if y.shape != common_shape:
        by = Expand.apply(y, shape=common_shape)

    return bx, by

# Step 42 - bind_binary_tensor_methods
def bind_binary_tensor_methods():
    # Attach broadcasting binary operations to the Tensor class.
    def add(self, other):
        x, y = broadcasted(self, other)
        return Add.apply(x, y)

    def sub(self, other):
        x, y = broadcasted(self, other)
        return Sub.apply(x, y)

    def mul(self, other):
        x, y = broadcasted(self, other)
        return Mul.apply(x, y)

    def div(self, other):
        x, y = broadcasted(self, other)
        return Div.apply(x, y)

    Tensor.add = add
    Tensor.sub = sub
    Tensor.mul = mul
    Tensor.div = div

    Tensor.__add__ = add
    Tensor.__sub__ = sub
    Tensor.__mul__ = mul
    Tensor.__truediv__ = div

# Step 43 - bind_movement_tensor_methods
def bind_movement_tensor_methods():
    # Build the differentiable Expand Function.
    Expand = type(
        "Expand",
        (Function,),
        {
            "forward": expand_function_forward,
            "backward": expand_function_backward,
        },
    )

    # Build the differentiable Permute Function.
    permute_forward, permute_backward = permute_function_forward_backward()
    Permute = type(
        "Permute",
        (Function,),
        {
            "forward": permute_forward,
            "backward": permute_backward,
        },
    )

    def reshape(self, *args):
        return Reshape.apply(self, shape=args[0])

    def expand_method(self, *args):
        return Expand.apply(self, shape=args[0])

    def permute_method(self, *args):
        return Permute.apply(self, order=args[0])

    return {
        "reshape": reshape,
        "expand": expand_method,
        "permute": permute_method,
    }

# Step 44 - bind_reduce_tensor_methods
def bind_reduce_tensor_methods():
    def normalize_axis(axis, ndim):
        if axis is None:
            return tuple(range(ndim))

        if isinstance(axis, int):
            axis = (axis,)

        return tuple(a + ndim if a < 0 else a for a in axis)

    def reduced_shape(shape, axes):
        axes = set(axes)
        return tuple(
            dim for i, dim in enumerate(shape)
            if i not in axes
        )

    def sum_method(self, axis=None, keepdim=False):
        axes = normalize_axis(axis, len(self.shape))

        if axis is None:
            reduce_axis = None
        elif len(axes) == 1:
            reduce_axis = axes[0]
        else:
            reduce_axis = axes

        out = Sum.apply(self, axis=reduce_axis)

        if not keepdim:
            new_shape = reduced_shape(out.shape, axes)
            out = Reshape.apply(out, shape=new_shape)

        return out

    def max_method(self, axis=None, keepdim=False):
        axes = normalize_axis(axis, len(self.shape))

        if axis is None:
            reduce_axis = None
        elif len(axes) == 1:
            reduce_axis = axes[0]
        else:
            reduce_axis = axes

        out = Max.apply(self, axis=reduce_axis)

        if not keepdim:
            new_shape = reduced_shape(out.shape, axes)
            out = Reshape.apply(out, shape=new_shape)

        return out

    Tensor.sum = sum_method
    Tensor.max = max_method

# Step 45 - tensor_mean
def tensor_mean(x, axis=None, keepdim=False):
    # Normalize the reduction axes.
    if axis is None:
        axes = tuple(range(len(x.shape)))
        reduce_axis = None
    elif isinstance(axis, int):
        normalized = axis + len(x.shape) if axis < 0 else axis
        axes = (normalized,)
        reduce_axis = normalized
    else:
        axes = tuple(
            a + len(x.shape) if a < 0 else a
            for a in axis
        )
        reduce_axis = axes

    # Perform the differentiable sum reduction.
    summed = Sum.apply(x, axis=reduce_axis)

    # Number of elements reduced.
    count = 1
    for ax in axes:
        count *= x.shape[ax]

    # Remove reduced dimensions when keepdim=False.
    if not keepdim:
        result_shape = tuple(
            dim for i, dim in enumerate(x.shape)
            if i not in set(axes)
        )
        summed = Reshape.apply(summed, shape=result_shape)

    # Divide by the number of reduced elements.
    divisor = Tensor(
        LazyBuffer.const(float(count), summed.shape),
        requires_grad=False,
    )

    return Div.apply(summed, divisor)

# Step 46 - tensor_transpose
def tensor_transpose(x, ax1=-2, ax2=-1):
    # Resolve negative axes relative to the tensor's number of dimensions.
    ndim = len(x.shape)

    if ax1 < 0:
        ax1 += ndim
    if ax2 < 0:
        ax2 += ndim

    # Build the permutation that swaps the two requested axes.
    order = list(range(ndim))
    order[ax1], order[ax2] = order[ax2], order[ax1]

    # Reuse the existing differentiable permute machinery.
    permute_fn = bind_movement_tensor_methods()["permute"]
    return permute_fn(x, tuple(order))

# Step 47 - tensor_matmul_2d
def tensor_matmul_2d(a, b):
    # a: (m, k), b: (k, n)
    m, k = a.shape
    k2, n = b.shape

    if k != k2:
        raise ValueError(
            f"Incompatible shapes for matmul: {a.shape} and {b.shape}"
        )

    # Reshape while preserving autograd.
    a = Reshape.apply(a, shape=(m, k, 1))
    b = Reshape.apply(b, shape=(1, k, n))

    # Expand while preserving autograd.
    methods = bind_movement_tensor_methods()
    expand_fn = methods["expand"]

    a = expand_fn(a, (m, k, n))
    b = expand_fn(b, (m, k, n))

    # Elementwise multiplication and reduction over k.
    product = Mul.apply(a, b)
    result = Sum.apply(product, axis=1)

    # Remove the kept reduction dimension.
    return Reshape.apply(result, shape=(m, n))

# Step 48 - tensor_softmax
def tensor_softmax(x, axis=-1):
    # Normalize negative axis indices.
    if axis < 0:
        axis += len(x.shape)

    # Keep the reduced dimension so broadcasting works naturally.
    max_val = Max.apply(x, axis=axis)

    # Subtract the maximum for numerical stability.
    shifted = Sub.apply(x, broadcasted(x, max_val)[1])

    # Exponentiate the stabilized logits.
    exp_values = Exp.apply(shifted)

    # Sum along the softmax axis, keeping the dimension.
    total = Sum.apply(exp_values, axis=axis)

    # Divide by the normalization constant.
    exp_values, total = broadcasted(exp_values, total)

    return Div.apply(exp_values, total)

# Step 49 - tensor_log_softmax
def tensor_log_softmax(x, axis=-1):
    # Normalize negative axis indices.
    if axis < 0:
        axis += len(x.shape)

    # Compute max along the requested axis, keeping the dimension.
    max_val = Max.apply(x, axis=axis)

    # Shift logits for numerical stability.
    x_shifted = Sub.apply(
        x,
        broadcasted(x, max_val)[1],
    )

    # Compute log(sum(exp(x - max))).
    exp_values = Exp.apply(x_shifted)
    sum_exp = Sum.apply(exp_values, axis=axis)
    log_sum_exp = Log.apply(sum_exp)

    # log_softmax = x - max(x) - log(sum(exp(x - max(x)))).
    x_shifted, log_sum_exp = broadcasted(x_shifted, log_sum_exp)

    result = Sub.apply(x_shifted, log_sum_exp)

    # Promote the final result so rounded values are represented cleanly.
    result.lazydata = LazyBuffer(result.lazydata._np.astype(np.float64))

    return result

# Step 50 - sparse_categorical_cross_entropy
def sparse_categorical_cross_entropy(logits, labels):
    # Accept either a Tensor or ordinary Python / NumPy data.
    if not isinstance(logits, Tensor):
        logits = tensor_from_data(logits, requires_grad=False)

    # Compute stable log-probabilities over the class dimension.
    log_probs = tensor_log_softmax(logits, axis=-1)

    n, c = logits.shape

    # Create a one-hot mask for the provided class labels.
    one_hot = np.zeros((n, c), dtype=np.float32)
    for i, label in enumerate(labels):
        one_hot[i, int(label)] = 1.0

    target_mask = tensor_from_data(one_hot, requires_grad=False)

    # Select the log-probability corresponding to each target class.
    selected = Mul.apply(log_probs, target_mask)

    # Sum over classes to get one value per sample.
    per_sample = Sum.apply(selected, axis=1)

    # Sum over the batch.
    total = Sum.apply(per_sample, axis=0)

    # Divide by the batch size.
    divisor = tensor_from_data(float(n), requires_grad=False)
    loss = Div.apply(total, divisor)

    # Negate the mean log-probability.
    negative_one = tensor_from_data(-1.0, requires_grad=False)
    loss = Mul.apply(loss, negative_one)

    # Convert the final (1, 1) result into a true scalar Tensor.
    return Reshape.apply(loss, shape=())

# Step 51 - Linear
class Linear:
    # Build a fully connected layer: x @ W + b.
    def __init__(self, in_features, out_features, seed=None):
        self.weight = tensor_randn(
            (in_features, out_features),
            seed=seed,
            requires_grad=True,
        )

        self.bias = tensor_randn(
            (out_features,),
            seed=None if seed is None else seed + 1,
            requires_grad=True,
        )

    def __call__(self, x):
        # Compute x @ W.
        out = tensor_matmul_2d(x, self.weight)

        # Broadcast the bias across the batch dimension and add it.
        out, bias = broadcasted(out, self.bias)
        return Add.apply(out, bias)

    def parameters(self):
        # Return the trainable tensors owned by this layer.
        return [self.weight, self.bias]

# Step 52 - MLP
class MLP:
    """Two-layer MLP: Linear -> relu -> Linear."""
    def __init__(self, in_features, hidden, out_features, seed=None):
        # Build the two Linear layers.
        self.layer1 = Linear(in_features, hidden, seed=seed)
        self.layer2 = Linear(
            hidden,
            out_features,
            seed=None if seed is None else seed + 1,
        )

        # Get the bound ReLU operation.
        self._relu = bind_unary_tensor_methods()["relu"]

    def __call__(self, x):
        # Accept either a Tensor or array-like input.
        if not isinstance(x, Tensor):
            x = tensor_from_data(x)

        # Linear -> ReLU -> Linear.
        x = self.layer1(x)
        x = self._relu(x)
        x = self.layer2(x)

        return x

    def parameters(self):
        # Return all trainable parameters from both layers.
        return self.layer1.parameters() + self.layer2.parameters()

# Step 53 - sgd_step
def sgd_step(parameters, learning_rate):
    # Update each parameter in place using its gradient.
    _, BinaryOps, _, _ = make_op_enums()

    for parameter in parameters:
        if parameter.grad is None:
            continue

        lr = LazyBuffer.const(
            float(learning_rate),
            parameter.shape,
        )

        scaled_grad = lazybuffer_binary_e(
            parameter.grad.data,
            BinaryOps.MUL,
            lr,
        )

        updated = lazybuffer_binary_e(
            parameter.data,
            BinaryOps.SUB,
            scaled_grad,
        )

        parameter.data = updated

    return None

# Step 54 - zero_grad
def zero_grad(parameters):
    # Reset each parameter's gradient so the next backward pass starts fresh.
    for parameter in parameters:
        parameter.grad = None

    return None

# Step 55 - make_toy_digit_dataset
def make_toy_digit_dataset(num_samples, seed=0):
    # Define the three flattened 3x3 binary digit prototypes.
    prototypes = np.array(
        [
            [0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 1, 1, 0, 1, 0, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 0, 1],
        ],
        dtype=np.float32,
    )

    # Use RandomState so the dataset is reproducible.
    rng = np.random.RandomState(seed)

    # Draw labels before drawing the noise, as required.
    y = rng.randint(0, 3, size=num_samples)

    # Generate Gaussian noise and add the corresponding prototype.
    noise = (0.1 * rng.randn(num_samples, 9)).astype(np.float32)
    X = prototypes[y] + noise

    return X.astype(np.float32), y.astype(np.int64)

# Step 56 - accuracy
def accuracy(logits, labels):
    # Accept either a Tensor or a raw NumPy array.
    if isinstance(logits, Tensor):
        values = logits.numpy()
    else:
        values = np.asarray(logits)

    # Predict the class with the highest logit for each sample.
    predictions = np.argmax(values, axis=1)

    # Return the fraction of correct predictions as a Python float.
    return float(np.mean(predictions == np.asarray(labels)))

# Step 57 - train_mlp
def train_mlp(X, y, epochs=50, learning_rate=0.1, hidden=16, seed=0):
    # Convert the training data to arrays so shapes and class counts are available.
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    # Infer the input dimension and number of output classes from the data.
    in_features = X.shape[1]
    out_features = int(np.max(y)) + 1

    # Build the model.
    model = MLP(
        in_features,
        hidden,
        out_features,
        seed=seed,
    )

    parameters = model.parameters()
    loss_history = []

    # Full-batch gradient descent.
    for _ in range(epochs):
        # Clear gradients from the previous iteration.
        zero_grad(parameters)

        # Forward pass.
        logits = model(X)

        # Compute the classification loss.
        loss = sparse_categorical_cross_entropy(logits, y)

        # Record the scalar loss value.
        loss_history.append(float(loss.numpy()))

        # Reverse-mode autodiff.
        tensor_backward(loss)

        # Update model parameters.
        sgd_step(parameters, learning_rate)

    return model, loss_history

# Step 58 - evaluate_mlp (not yet solved)
# TODO: implement

