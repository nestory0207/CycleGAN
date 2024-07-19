import os, pdb
import nvidia_smi

class DeviceController():
    def init_shutdown(func):
        """
        nvml을 initila, shutdown해주는 decorater
        """
        def wrapper(*args, **kwargs):
            nvidia_smi.nvmlInit()
            value = func(*args, **kwargs)
            nvidia_smi.nvmlShutdown()
            return value
        return wrapper
            
    @init_shutdown
    def check_usage(memory_limit):
        """
        GPU 메모리 사용량을 확인하여 여유가 있는 GPU를 담아주는 함수.
            Args:
                - memory_percentage: GPU의 사용가능한 메모리 비율을 조절하는 파라미터
            Return:
                - available_gpus_num_list: 사용가능한 GPU device의 number를 담는 리스트
                - create_virtual: device의 총 메모리를 확인하고 설정한 메모리의 배수보다 작은지 확인하여 virtual GPU를 생성여부 return
        
        """
        deviceCount = nvidia_smi.nvmlDeviceGetCount()
        available_gpus_num_list = [] # GPUs has over 90% free memory
        for i in range(deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))

            total_memory_gb = info.total/1e9
            available_memory_gb = info.free/1e9
            memory_limit_gb = memory_limit/1e3
            
            if total_memory_gb<2*memory_limit_gb:
                create_virtual = False
            else:
                create_virtual = True
            
            if available_memory_gb>memory_limit_gb: # if available memory > 12GB
                available_gpus_num_list.append(i)
        
        if len(available_gpus_num_list)==0:
            raise ValueError("There is no available gpu devices")
            
        return available_gpus_num_list, create_virtual
    
def set_cuda_visible(gpu_num):
    available_gpus, _ = DeviceController.check_usage(memory_limit=10*1024)
    available_gpus = ','.join([str(i) for i in available_gpus])
    available_gpus = available_gpus[:gpu_num]
    
    os.environ['CUDA_VISIBLE_DEVICES'] = available_gpus
    print("Set CUDA visible devices!")
    print(f"GPU:{available_gpus}")