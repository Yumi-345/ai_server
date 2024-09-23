Author = "xuwj"

import ctypes
import cupy as cp
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import time
import pyds
import threading


TIME_OUT = 3

class SafeLock:
    def __init__(self, time_out=10):
        self.lock = threading.Lock()
        self.lock_held = threading.Event()
        self.lock_flag = False
        self.time_out = time_out
        self.lock_time = time.time()
        t = threading.Thread(target=self.monitor_lock)
        t.setDaemon(True)
        t.start()

    def monitor_lock(self,):
        while True:
            if time.time() - self.lock_time < self.time_out:
                time.sleep(0.5)
            elif self.lock_held.is_set():
                    print(f">>>锁超时{self.time_out}, 手动释放====")
                    self.release()
            else:
                time.sleep(self.time_out/2)

    def acquire(self):
        self.lock.acquire()
        self.lock_time = time.time()
        self.lock_held.set()

    def release(self):
        if self.lock_held.is_set():
            try:
                self.lock.release()
                self.lock_held.clear()
                # print(">>>锁释放=======")
            except Exception as e:
                print(f"******************error in release lock***************:\n{e}")


def check_pipeline_elements(pipeline):
    for element in pipeline.iterate_elements():
        state_change_return, current, pending = element.get_state(1 * Gst.SECOND)
        print(f"Element {element.get_name()} state: {current}, Pending state: {pending}")
        if state_change_return == Gst.StateChangeReturn.FAILURE:
            print(f"Element {element.get_name()} failed to change state.")
            return False
    return True


def stop_pipeline(*args):
    for pipeline in args:
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

                if time.time() - start_time > TIME_OUT:
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

                if time.time() - start_time > TIME_OUT:
                    print("Failed to ready pipeline within the timeout period.")
                    break

            pipeline.set_state(Gst.State.NULL)
            while True:
                state_change_return, current, pending = pipeline.get_state(1 * Gst.SECOND)
                # print(f"Current pipeline state: {current}, Pending state: {pending}")
                if current == Gst.State.NULL:
                    print("Pipeline stopped successfully.")
                    break
                            
                if state_change_return == Gst.StateChangeReturn.ASYNC:
                    print("Pipeline stopped async")
                    break

                if time.time() - start_time > TIME_OUT:
                    print("Failed to stop pipeline within the timeout period.")
                    if not check_pipeline_elements(pipeline):
                        print("One or more elements failed to change state.")
                    break

# Function to start the pipeline
def start_pipeline(*args):
    for pipeline in args:
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

                if time.time() - start_time > TIME_OUT:
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

                if time.time() - start_time > TIME_OUT:
                    print("Failed to pause pipeline within the timeout period.")
                    break
            
            pipeline.set_state(Gst.State.PLAYING)
            while True:
                state_change_return, current, pending = pipeline.get_state(1 * Gst.SECOND)
                # print(f"Current pipeline state: {current}, Pending state: {pending}")
                if current == Gst.State.PLAYING:
                    print("Pipeline started successfully.")
                    break
                
                if state_change_return == Gst.StateChangeReturn.ASYNC:
                    print("Pipeline start async")
                    # break

                if time.time() - start_time > TIME_OUT:
                    print("Failed to start pipeline within the timeout period.")
                    if not check_pipeline_elements(pipeline):
                        print("One or more elements failed to change state.")
                    break



def remove_element(pipeline, element):
    # 将元素的状态设置为 NULL，停止元素的操作
    element.set_state(Gst.State.NULL)
    
    # 处理动态 pad 的解绑
    for pad in element.pads:
        if pad.is_linked():
            peer_pad = pad.get_peer()
            pad.unlink(peer_pad)
    
    # 从管道中移除元素
    pipeline.remove(element)
    
    # 释放 Python 对象的引用
    del element


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

    # 下边为创建cuda流，若使用null，则多线程变为单线程运行，但是提示信息表示不使用null，可能存在非法访问cuda内存问题，目前未遇到，待定
    stream = cp.cuda.stream.Stream() # Use null stream to prevent other cuda applications from making illegal memory access of buffer
    # Modify the red channel to add blue tint to image
    
    with stream:
        n_frame_cpu = n_frame_gpu.get()
    return n_frame_cpu
