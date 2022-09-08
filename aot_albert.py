from turtle import forward
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, ReformerConfig, BigBirdConfig, BertConfig
import transformers
import torch
from functorch.compile import memory_efficient_fusion, aot_module, draw_graph_compile, nop, ts_compile, tensorexpr_compile, min_cut_rematerialization_partition
import torch.utils._pytree as pytree
import time
from torch import optim
import torch.nn as nn
from torch.nn.utils import _stateless
import pandas as pd
from functorch.compile import compiled_module, tvm_compile
# import intel_extension_for_pytorch as ipex

pytree._register_pytree_node(transformers.modeling_outputs.MaskedLMOutput, lambda x: ([x.loss, x.logits], None), lambda values, _: transformers.modeling_outputs.MaskedLMOutput(loss=values[0], logits=values[1]))

pytree._register_pytree_node(transformers.modeling_outputs.Seq2SeqLMOutput, lambda x: ([x.loss, x.logits], None), lambda values, _: transformers.modeling_outputs.Seq2SeqLMOutput(loss=values[0], logits=values[1]))

pytree._register_pytree_node(transformers.modeling_outputs.CausalLMOutputWithCrossAttentions, lambda x: ([x.loss, x.logits], None), lambda values, _: transformers.modeling_outputs.CausalLMOutputWithCrossAttentions(loss=values[0], logits=values[1]))

pytree._register_pytree_node(transformers.models.longformer.modeling_longformer.LongformerMaskedLMOutput, lambda x: ([x.loss, x.logits], None), lambda values, _: transformers.models.longformer.modeling_longformer.LongformerMaskedLMOutput(loss=values[0], logits=values[1]))

torch.manual_seed(42)
benchmarks = [
    #(AutoConfig.from_pretrained("albert-base-v2"), AutoModelForMaskedLM, (8, 512), []),
    #(AutoConfig.from_pretrained("gpt2"), AutoModelForCausalLM, (4, 512), []),
    (BertConfig(), AutoModelForMaskedLM, (4, 512), []),
    # (AutoConfig.from_pretrained("facebook/bert-base"), AutoModelForSeq2SeqLM, (4, 512), []), # Doesn't work with nn.utils._stateless for some reason...
    # (ReformerConfig(), AutoModelForMaskedLM, (8, 4096), []), # not sure...
    # (BigBirdConfig(attention_type="block_sparse"), AutoModelForMaskedLM, (2, 1024), []), # not sure...
    # (AutoConfig.from_pretrained("distilbert-base-uncased"),  AutoModelForMaskedLM, (8, 512), []), # encounters inf as a global value
]

class base_model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.linear = nn.Linear(3, 3)
        # self.relu = nn.ReLU()

    # def forward(self, x):
    # # def forward(self, x, y):
    def forward(self, a, b, c, d):
        # z = torch.add(x, 1)
        # z = self.relu(z)
        x = a + b + c + d
        # x = self.linear(x)
        # x.add_(1)
        # return (x + y).sum()
        # return z
        return x.cos().cos()


device = 'cpu'
# fw_compiler = nop
# bw_compiler = nop
fw_compiler = ts_compile
bw_compiler = ts_compile
# fw_compiler = tensorexpr_compile
# bw_compiler = tensorexpr_compile
numerical_diffs = []
results = []
for config, model_type, input_size, not_supported_dtypes in benchmarks:
    print("tanglei 1")
    for dtype in [torch.float]:
        print("tanglei 2")
        if dtype in not_supported_dtypes:
            continue
        for attr in dir(config):
            if 'drop' in attr:
                setattr(config, attr, 1e-60) # So we can check for correct gradients without eliminating the dropout computation
        model = model_type.from_config(config).to(device, dtype=dtype)
        input_ids = torch.randint(0, config.vocab_size, input_size).to(device)
        decoder_ids = torch.randint(0, config.vocab_size, input_size).to(device)


        base_model = base_model()
        train_inputs = {'input_ids': input_ids, 'labels': decoder_ids}

        
        def bench_model(name, mod):
            m = None
            for i in range(5):
                print("tanglei bench_model 1")
                out = mod(**train_inputs).loss.abs().sum()
                print("tanglei bench_model 2")
                out.backward()
                print("tanglei bench_model 3")
            iters = 20
            begin = time.time()
            for _ in range(iters):
                mod(**train_inputs).loss.sum().backward()
            t = (time.time()-begin)/iters
            print(name, (time.time()-begin)/iters)
            return t, m

        #import pdb
        #pdb.set_trace() 
        print("tanglei 3")
        torch._C._jit_set_texpr_reductions_enabled(True)
        # print(model)
        # exit()
        aot_model = compiled_module(model, fw_compiler, bw_compiler) #memory_efficient_fusion(model)
        # aot_base_model = compiled_module(base_model, fw_compiler, bw_compiler, partition_fn=min_cut_rematerialization_partition) #memory_efficient_fusion(model)
        # aot_base_model = compiled_module(base_model, fw_compiler, bw_compiler) #memory_efficient_fusion(model)
        print("tanglei 4")
        model_name = type(model).__name__
        print("tanglei 5")
        # t,m = bench_model("eager", model)
        print("tanglei 6")


        ## Checking correctness
        def clear_params(model):
            for i in model.parameters(): i.grad = None

        # clear_params(aot_model)
        torch.manual_seed(0)
        print("tanglei 7")
        # for _ in range(10):
        out2 = aot_model(**train_inputs).loss
        # a, b, c, d = [torch.randn(3, 3, 3, requires_grad=True) for _ in range(4)]
        # base_out2 = aot_base_model(torch.randn(5, 5)).loss
        # base_out2 = aot_base_model(a, b, c, d)
        # base_out2 = aot_base_model(a)
        # base_out2 = aot_base_model(torch.randn(5, 5))
        print("tanglei 8")
        out2.sum().backward()
        # base_out2.sum().backward()
        print("tanglei 9")
        # grad2 = [i.grad for i in aot_model.parameters()]
        # base_grad2 = [i.grad for i in aot_base_model.parameters()]

        # for _ in range(10):
        #     # base_out2 = aot_base_model(a, b, c, d)
        #     # base_out2.sum().backward()
        #     out2 = aot_model(**train_inputs).loss
        #     out2.sum().backward()
# Lets write a function to benchmark the forward and backward pass
import time
import statistics

def bench(fn, args, prefix):
    warmup = 10
    # iterations = 100
    iterations = 10

    for _ in range(warmup):
        ref = fn(**args).loss
        ref.sum().backward()
    
    fw_latencies = []
    bw_latencies = []
    for _ in range(iterations):
        # for arg in args:
        #     arg.grad = None

        fw_begin = time.perf_counter()
        ref = fn(**args)
        fw_end = time.perf_counter()

        loss = ref.loss.sum() 

        bw_begin = time.perf_counter()
        loss.backward()
        bw_end = time.perf_counter()

        fw_latencies.append(fw_end - fw_begin)
        bw_latencies.append(bw_end - bw_begin)
    
    avg_fw_latency = statistics.mean(fw_latencies) * 10**6
    avg_bw_latency = statistics.mean(bw_latencies) * 10**6
    print(prefix, "Fwd = " + str(avg_fw_latency) + " us", "Bwd = " + str(avg_bw_latency) + " us", sep=', ')

inputs = [torch.randn(1024, 2048, requires_grad=True) for _ in range(4)]
cloned_inputs = [x.clone().detach().requires_grad_(True) for x in inputs]
cloned_a, cloned_b, cloned_c, cloned_d = cloned_inputs
# Benchmark the Eager and AOT Autograd functions
# bench(base_model, inputs, "Eager")
bench(model, train_inputs, "Eager")
# bench(aot_base_model, inputs, "AOT")
bench(aot_model, train_inputs, "AOT")
# ref = model(**train_inputs)

# # res = aot_model(*cloned_inputs)
# res = aot_model(**train_inputs)
# loss = res.sum()
# loss.backward()
# assert torch.allclose(ref, res)
# # print(ref)
# # print(res)
# assert torch.allclose(inputs[0].grad, cloned_a.grad)
# # print(inputs[0].grad)
# # print(cloned_a.grad)
# assert torch.allclose(inputs[1].grad, cloned_b.grad)
# # print(inputs[1].grad)
# # print(cloned_b.grad)
# assert torch.allclose(inputs[2].grad, cloned_c.grad)
# # print(inputs[2].grad)
# # print(cloned_c.grad)
# assert torch.allclose(inputs[3].grad, cloned_d.grad)
# print(inputs[3].grad)
# print(cloned_d.grad)
#         # clear_params(model)
#         clear_params(base_model)
#         torch.manual_seed(0)
#         print("tanglei 10")
#         out1 = model(**train_inputs).loss
#         print("tanglei 11")
#         out1.sum().backward()
#         print("tanglei 12")
#         grad1 = [i.grad for i in model.parameters()]

#         if model_name == 'LongformerForMaskedLM': # Longformer seems to have worse precision
#             atol = 5e-3
#             rtol = 1e-3
#         elif dtype == torch.float:
#             atol = 1e-4
#             rtol = 5e-3
#         else:
#             atol = 1e-2
#             rtol = 1e-1
#         try:
#             torch.testing.assert_close(out2, out1, atol=atol, rtol=rtol)
#             torch.testing.assert_close(grad2, grad1, atol=atol, rtol=rtol)
#         except AssertionError as e:
#             print(e)
#             numerical_diffs.append((model_name, str(dtype), e))
#         print()

# for model_name, dtype, err in numerical_diffs:
#     print(f"Numerical differences in {model_name} - {dtype} found")
#     print(err)
#     print()

# print(pd.DataFrame(results).to_markdown(index=False, floatfmt=".3f"))