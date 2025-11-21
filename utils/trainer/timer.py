import time
import datetime
import torch

class TimingContext:
    def __init__(self, name="Block"):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        torch.cuda.synchronize() # 如果用 GPU，必须同步才能测准
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        print(f"[{self.name}] elapsed: {(time.time() - self.start)*1000:.2f} ms")

class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.epoch_start_time = 0
    
    def start_epoch(self):
        self.epoch_start_time = time.time()
        
    def end_epoch(self) -> str:
        elapsed = time.time() - self.epoch_start_time
        return str(datetime.timedelta(seconds=int(elapsed)))
    
    def total_time(self) -> str:
        elapsed = time.time() - self.start_time
        return str(datetime.timedelta(seconds=int(elapsed)))
