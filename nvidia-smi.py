import torch
import time
while(True):
    time.sleep(1)
    for i in range(4):
        print('{} {}: usage {}%'.format(i, torch.cuda.get_device_name(i), torch.cuda.memory_usage(i)))
    print("\n")
