import pyds
import ctypes
import cupy as cp
import time

class BaseServer():
    def __init__(self, ):
        pass

    def create_pipeline(self, ):
        pass 

    def get_data_GPU(self, gst_buffer, frame_meta):
        # Create dummy owner object to keep memory for the image array alive
        owner = None
        # Getting Image data using nvbufsurface
        # the input should be address of buffer and batch_id
        # Retrieve dtype, shape of the array, strides, pointer to the GPU buffer, and size of the allocated memory
        data_type, shape, strides, dataptr, size = pyds.get_nvds_buf_surface_gpu(hash(gst_buffer), frame_meta.batch_id)
        # dataptr is of type PyCapsule -> Use ctypes to retrieve the pointer as an int to pass into cupy
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        # Get pointer to buffer and create UnownedMemory object from the gpu buffer
        c_data_ptr = ctypes.pythonapi.PyCapsule_GetPointer(dataptr, None)
        unownedmem = cp.cuda.UnownedMemory(c_data_ptr, size, owner)
        # Create MemoryPointer object from unownedmem, at index 0
        memptr = cp.cuda.MemoryPointer(unownedmem, 0)
        # Create cupy array to access the image data. This array is in GPU buffer
        n_frame_gpu = cp.ndarray(shape=shape, dtype=data_type, memptr=memptr, strides=strides, order='C')
        # Initialize cuda.stream object for stream synchronization
        stream = cp.cuda.stream.Stream(null=True) # Use null stream to prevent other cuda applications from making illegal memory access of buffer
        # Modify the red channel to add blue tint to image
        
        with stream:
            n_frame_cpu = n_frame_gpu.get()
        # stream.synchronize()
        return n_frame_cpu

