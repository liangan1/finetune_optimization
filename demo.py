import torch
import torch.nn as nn
from torch import optim

from functorch.compile import memory_efficient_fusion, aot_module, draw_graph_compile, nop, ts_compile, min_cut_rematerialization_partition, compiled_module, tvm_compile
# from functorch.compile import tensorexpr_compile
import torch.utils._pytree as pytree

import time
import statistics


torch.manual_seed(0)
class base_model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b, c, d):
        x = a + b + c + d
        return x.cos().cos()


# device = 'cpu'
# fw_compiler = nop
# bw_compiler = nop
fw_compiler = ts_compile
bw_compiler = ts_compile
# fw_compiler = tensorexpr_compile
# bw_compiler = tensorexpr_compile

base_model = base_model()
aot_base_model = compiled_module(base_model, fw_compiler, bw_compiler) #memory_efficient_fusion(model)

def bench(fn, args, prefix):
    warmup = 10
    iterations = 100

    for _ in range(warmup):
        ref = fn(*args)
        ref.sum().backward()
    
    fw_latencies = []
    bw_latencies = []
    for _ in range(iterations):
        for arg in args:
            arg.grad = None

        fw_begin = time.perf_counter()
        ref = fn(*args)
        fw_end = time.perf_counter()

        loss = ref.sum() 

        bw_begin = time.perf_counter()
        loss.backward()
        bw_end = time.perf_counter()

        fw_latencies.append(fw_end - fw_begin)
        bw_latencies.append(bw_end - bw_begin)
    
    avg_fw_latency = statistics.mean(fw_latencies) * 10**6
    avg_bw_latency = statistics.mean(bw_latencies) * 10**6
    print(prefix, "Fwd = " + str(avg_fw_latency) + " us", "Bwd = " + str(avg_bw_latency) + " us", sep=', ')

def main():
    inputs = [torch.randn(1024, 2048, requires_grad=True) for _ in range(4)]
    cloned_inputs = [x.clone().detach().requires_grad_(True) for x in inputs]
    cloned_a, cloned_b, cloned_c, cloned_d = cloned_inputs
    # Benchmark the Eager and AOT Autograd functions
    bench(base_model, inputs, "Eager")
    bench(aot_base_model, inputs, "AOT")
    ref = base_model(*inputs)

    res = aot_base_model(*cloned_inputs)
    loss = res.sum()
    loss.backward()
    assert torch.allclose(ref, res)
    assert torch.allclose(inputs[0].grad, cloned_a.grad)
    assert torch.allclose(inputs[1].grad, cloned_b.grad)
    assert torch.allclose(inputs[2].grad, cloned_c.grad)
    assert torch.allclose(inputs[3].grad, cloned_d.grad)

if __name__ == '__main__':
    main()