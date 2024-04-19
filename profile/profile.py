import torch
from torch.profiler import profile, ProfilerActivity
from baselines.rmsnorm import MyRMSNorm

# wrap all the code above into print statements printing a string to describe the output
print("cuda is available: ", torch.cuda.is_available())
print("number of cuda devices: ", torch.cuda.device_count())
print("current cuda device: ", torch.cuda.current_device())
print("cuda device name: ", torch.cuda.get_device_name(0))
print("cuda memory allocated: ", torch.cuda.memory_allocated())

# trace_handler is called every time a new trace becomes available
def cpu_trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
def gpu_trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    # prof.export_chrome_trace("prof/test_trace_" + str(prof.step_num) + ".json")
    

with torch.profiler.profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ],

    # In this example with wait=1, warmup=1, active=2, repeat=1,
    # profiler will skip the first step/iteration,
    # start warming up on the second, record
    # the third and the forth iterations,
    # after which the trace will become available
    # and on_trace_ready (when set) is called;
    # the cycle repeats starting with the next step

    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2,
        repeat=1),
    # [pick 1] For Flamegraphs:
    # on_trace_ready=cpu_trace_handler
    on_trace_ready=gpu_trace_handler
    # For Tensorboard:
    # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    ) as p:
        for iter in range(10):
            # torch.square(torch.randn(10000, 10000).cuda())
            # MyRMSNorm(dim=4096).forward(torch.randn(4096))
            MyRMSNorm(dim=4096).cuda().forward(torch.randn(4096).cuda())
            # send a signal to the profiler that the next iteration has started
            p.step()