import torch
from torch.profiler import profile, ProfilerActivity
from baselines.rmsnorm import MyRMSNorm

# wrap all the code above into print statements printing a string to describe the output
print("cuda is available: ", torch.cuda.is_available())
print("number of cuda devices: ", torch.cuda.device_count())
print("current cuda device: ", torch.cuda.current_device())
print("cuda device name: ", torch.cuda.get_device_name(0))
print("cuda memory allocated: ", torch.cuda.memory_allocated())

class Profiler:
    def __init__(self, device, operation_class, input_dims, export_chrome_trace=False):
        self.device = device
        self.operation_class = operation_class
        self.input_dims = input_dims

    def cpu_trace_handler(self, prof, export_chrome_trace=False):
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
        if export_chrome_trace:
            prof.export_chrome_trace("prof/test_trace_" + str(prof.step_num) + ".json")

    def gpu_trace_handler(self, prof, export_chrome_trace=False):
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
        if export_chrome_trace:
            prof.export_chrome_trace("prof/test_trace_" + str(prof.step_num) + ".json")

    def profile(self, batch_size=1):

        trace_handler = self.cpu_trace_handler if self.device == 'cpu' else self.gpu_trace_handler

        activities = [ProfilerActivity.CPU]
        if self.device == 'cuda':
            activities.append(ProfilerActivity.CUDA)

        # In this example with wait=1, warmup=1, active=2, repeat=1,
        # profiler will skip the first step/iteration,
        # start warming up on the second, record
        # the third and the forth iterations,
        # after which the trace will become available
        # and on_trace_ready (when set) is called;
        # the cycle repeats starting with the next step
        with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=2,
            repeat=1),
            on_trace_ready=trace_handler,
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
        ) as p:
            for _ in range(10):
                operation = self.operation_class(*self.input_dims)
                if self.device == 'cuda':
                    operation = operation.cuda()
                # TODO: unclear if I should using rand in the profiling loop
                operation.forward(torch.randn(batch_size,*self.input_dims).to(self.device))
                p.step()


dim = 4096*1
batch_size = 1
profiler = Profiler('cuda', MyRMSNorm, (dim,))
profiler.profile(batch_size=batch_size)

profiler = Profiler('cpu', MyRMSNorm, (dim,))
profiler.profile(batch_size=batch_size)