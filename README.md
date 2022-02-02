# Accelerated computing with CUDA in Python

#### 
In this we will be looking at some basic commands for programming in CUDA for creating acceleratied appliccation. So lets see what is cuda.
 [CUDA](https://developer.nvidia.com/cuda-zone) is a parallel computing platform and programming model that enables dramatic increases in computing performance by harnessing the power of the GPU. 
 
## NUMBA
- [Numba](http://numba.pydata.org/) is an open source JIT compiler that translates a subset of Python and NumPy code into fast machine code.  Numba can be used to accelerate Python functions for the CPU, as well as for NVIDIA GPUs. 
- Numba is a just-in-time, type-specializing, function compiler for accelerating numerically-focused Python for either a CPU or GPU.
   - **function compiler** : Numba compiles Python functions, not entire applications, and not parts of functions. Numba does not replace your Python interpreter, but is just another Python module that can turn a function into a (usually) faster function.
   - **type-specializing** : Numba speeds up your function by generating a specialized implementation for the specific data types you are using. Python functions are designed to operate on generic data types, which makes them very flexible, but also very slow. In practice, you only will call a function with a small number of argument types, so Numba will generate a fast implementation for each set of types
   - **just-in-time**: Numba translates functions when they are first called. This ensures the compiler knows what argument types you will be using. This also allows Numba to be used interactively in a Jupyter notebook just as easily as a traditional application.
   - **numerically-focused** : Currently, Numba is focused on numerical data types, like int, float, and complex. There is very limited string processing support, and many string use cases are not going to work well on the GPU. To get best results with Numba, you will likely be using NumPy arrays.
- Basic syntax

```
from numba import jit
import math

# The Numba compiler is just a function
@jit     #function decorator
def add(x, y):
    return x + y
  ```
  - Numba cannot compile all python codes. ( eg. codes containing dictionaries etc.). But this will not create error, numba will detect this and will execute it normally called **object mode**. We can manually force to run in **non python mode**
 ```
 from numba import njit

@njit
def cannot_compile(x):
    return x['key']

cannot_compile(dict(key='value'))
```
The above code will result in an error

### Vectorize
With Numba you simply implement a scalar function to be performed on all the inputs, decorate it with @vectorize, and Numba will figure out the broadcast rules for you.
This is very similar to [numpy vectorize](https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html?highlight=vectorize#numpy.vectorize)
```
from numba import vectorize

@vectorize
def add_ten(num):
    return num + 10 # This scalar operation will be performed on each element

nums = np.arange(10)
add_ten(nums) # pass the whole array into the ufunc, it performs the operation on each element
```
#### **CUDA for GPU**
```
 @vectorize(['int64(int64, int64)'], target='cuda') # Type signature and target are required for the GPU
 def add_ufunc(x, y):
   return x + y

 add_ufunc(a, b)
```
#### GPU Memory
- As a convenience, Numba has been automatically transferring this data to the GPU for us so that it can be operated on by the GPU. With this implicit data transfer Numba, acting conservatively, will automatically transfer the data back to the CPU after processing. This is very time consuming.
- To counter this problem we make use of **CUDA Device Arrays**. Device arrays will not be automatically transfered back to the host after processing, and can be reused as we wish on the device before ultimately, and only if necessary, sending them, or parts of them, back to the host.
```
@vectorize(['float32(float32, float32)'], target='cuda')
def add_ufunc(x, y):
    return x + y

n = 100000
x = np.arange(n).astype(np.float32)
y = 2 * x

from numba import cuda

x_device = cuda.to_device(x)
y_device = cuda.to_device(y)

# To avoid output to copy back to CPU, we are defining an output array inside the device itself.
out_device = cuda.device_array(shape=(n,), dtype=np.float32)  # does not initialize the contents, like np.empty()

add_ufunc(x_device, y_device, out=out_device)

# If you want to get back the output from the device
out_host = out_device.copy_to_host()
print(out_host[:10])
```
### Custom CUDA Kernels in Python
Writing custom CUDA kernels, while more challenging than writing GPU accelerated ufuncs, provides developers with tremendous flexibility for the types of functions they can send to run in parallel on the GPU. While remaining purely in Python, the way we write CUDA kernels using Numba is very reminiscent of how developers write them in CUDA C/C++.
-Let us see a sample code to understand better.
```
from numba import cuda

# Note the use of an `out` array. CUDA kernels written with `@cuda.jit` do not return values,
# just like their C counterparts. Also, no explicit type signature is required with @cuda.jit
@cuda.jit
def add_kernel(x, y, out):
    
    # The actual values of the following CUDA-provided variables for thread and block indices,
    # like function parameters, are not known until the kernel is launched.
    
    # This calculation gives a unique thread index within the entire grid (see the slides above for more)
    idx = cuda.grid(1)          # 1 = one dimensional thread grid, returns a single value.
                                # This Numba-provided convenience function is equivalent to
                                # `cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x`

    # This thread will do the work on the data element with the same index as its own
    # unique index within the grid.
    out[idx] = x[idx] + y[idx]
    
    
 import numpy as np

n = 4096
x = np.arange(n).astype(np.int32) # [0...4095] on the host
y = np.ones_like(x)               # [1...1] on the host

d_x = cuda.to_device(x) # Copy of x on the device
d_y = cuda.to_device(y) # Copy of y on the device
d_out = cuda.device_array_like(d_x) # Like np.array_like, but for device arrays

# Because of how we wrote the kernel above, we need to have a 1 thread to one data element mapping,
# therefore we define the number of threads in the grid (128*32) to equal n (4096).
threads_per_block = 128
blocks_per_grid = 32


add_kernel[blocks_per_grid, threads_per_block](d_x, d_y, d_out)
cuda.synchronize()
print(d_out.copy_to_host()) # Should be [1...4096]
```
  
