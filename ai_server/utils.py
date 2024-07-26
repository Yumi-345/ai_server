Author = "xuwj"

import ctypes
import cupy as cp
import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import time
import pyds



def check_pipeline_elements(pipeline):
    for element in pipeline.iterate_elements():
        state_change_return, current, pending = element.get_state(1 * Gst.SECOND)
        # print(f"Element {element.get_name()} state: {current}, Pending state: {pending}")
        if state_change_return == Gst.StateChangeReturn.FAILURE:
            print(f"Element {element.get_name()} failed to change state.")
            return False
    return True


def stop_pipeline(pipeline, timeout=5):
    if pipeline:
        start_time = time.time()
        pipeline.set_state(Gst.State.PAUSED)
        while True:
            state_change_return, current, pending = pipeline.get_state(1 * Gst.SECOND)
            # print(f"Current pipeline state: {current}, Pending state: {pending}")
            if current == Gst.State.PAUSED:
                print("Pipeline paused successfully.")
                break
                        
            if state_change_return == Gst.StateChangeReturn.ASYNC:
                print("Pipeline paused async")
                break

            if time.time() - start_time > timeout:
                print("Failed to pause pipeline within the timeout period.")
                break

        pipeline.set_state(Gst.State.READY)
        while True:
            state_change_return, current, pending = pipeline.get_state(1 * Gst.SECOND)
            # print(f"Current pipeline state: {current}, Pending state: {pending}")
            if current == Gst.State.READY:
                print("Pipeline ready successfully.")
                break
            
            if state_change_return == Gst.StateChangeReturn.ASYNC:
                print("Pipeline ready async")
                break

            if time.time() - start_time > timeout:
                print("Failed to ready pipeline within the timeout period.")
                break

        pipeline.set_state(Gst.State.NULL)
        while True:
            state_change_return, current, pending = pipeline.get_state(1 * Gst.SECOND)
            # print(f"Current pipeline state: {current}, Pending state: {pending}")
            if current == Gst.State.NULL:
                print("Pipeline stopped successfully.")
                return True
                        
            if state_change_return == Gst.StateChangeReturn.ASYNC:
                print("Pipeline stopped async")
                return True

            if time.time() - start_time > timeout:
                print("Failed to stop pipeline within the timeout period.")
                return False

# Function to start the pipeline
def start_pipeline(pipeline, timeout=5):
    if pipeline:
        start_time = time.time()
        
        pipeline.set_state(Gst.State.READY)
        while True:
            state_change_return, current, pending = pipeline.get_state(1 * Gst.SECOND)
            # print(f"Current pipeline state: {current}, Pending state: {pending}")
            if current == Gst.State.READY:
                print("Pipeline ready successfully.")
                break
                                    
            if state_change_return == Gst.StateChangeReturn.ASYNC:
                print("Pipeline ready async")
                break

            if time.time() - start_time > timeout:
                print("Failed to ready pipeline within the timeout period.")
                break

        pipeline.set_state(Gst.State.PAUSED)
        while True:
            state_change_return, current, pending = pipeline.get_state(0.1 * Gst.SECOND)
            # print(f"Current pipeline state: {current}, Pending state: {pending}")
            if current == Gst.State.PAUSED:
                print("Pipeline paused successfully.")
                break
                                    
            if state_change_return == Gst.StateChangeReturn.ASYNC:
                print("Pipeline paused async")
                break

            if time.time() - start_time > timeout:
                print("Failed to pause pipeline within the timeout period.")
                break
        
        pipeline.set_state(Gst.State.PLAYING)
        while True:
            state_change_return, current, pending = pipeline.get_state(1 * Gst.SECOND)
            # print(f"Current pipeline state: {current}, Pending state: {pending}")
            if current == Gst.State.PLAYING:
                print("Pipeline started successfully.")
                return True
            
            if state_change_return == Gst.StateChangeReturn.ASYNC:
                print("Pipeline start async")
                # time.sleep(5)
                return True

            if time.time() - start_time > timeout:
                print("Failed to start pipeline within the timeout period.")
                if not check_pipeline_elements(pipeline):
                    print("One or more elements failed to change state.")
                return False


def remove_element_from_pipeline(pipeline, element, timeout=5):
    if pipeline and element:
        pipeline.remove(element)
        element.set_state(Gst.State.NULL)
        start_time = time.time()
        while True:
            state_change_return, current, pending = element.get_state(1 * Gst.SECOND)
            print(f"Current element state: {current}, Pending state: {pending}")
            if current == Gst.State.NULL:
                print(f"Element {element.get_name()} removed and set to NULL state.")
                break
                                    
            if state_change_return == Gst.StateChangeReturn.ASYNC:
                print(f"Element {element.get_name()} removed async and set to NULL state.")
                break

            if time.time() - start_time > timeout:
                print(f"Failed to set element {element.get_name()} to NULL state within the timeout period.")
                break


def get_data_GPU(gst_buffer, frame_meta):
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
    # stream = cp.cuda.stream.Stream(null=True) # Use null stream to prevent other cuda applications from making illegal memory access of buffer
    # Modify the red channel to add blue tint to image
    
    # with stream:
    return n_frame_gpu.get()
