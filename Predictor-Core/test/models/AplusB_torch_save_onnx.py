import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class AplusB_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return torch.add(A, B)




my_module = AplusB_Model()


# An example input you would normally provide to your model's forward() method.
example_A = torch.rand(3,2)
example_B = torch.rand(3,2)


# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_module = torch.jit.trace(my_module, (example_A, example_B))

output_traced = traced_module(torch.ones(2,2), 20*torch.ones(2,2))

print("output_traced: \n", output_traced)


# Use torch.jit.script to generate a scriptmodule without tracing.

script_module = torch.jit.script(my_module)

output_script = script_module(torch.tensor([1.,2.,3.]), torch.tensor([3.,4.,5.]))

print("output_script: \n", output_script)
# # Save Model
# sm.save("my_module_model.pt")

input_names = ['input_a', 'input_b']
output_names = ['result']

dynamic_axes = {
  'input_a': {0 : 'batch_size'},
  'input_b': {0 : 'batch_size'},
  'result': {0 : 'batch_size'},
}


batch_size = 100

dummy_A = torch.Tensor(batch_size, 2)
dummy_B = torch.Tensor(batch_size, 2)





torch.onnx.export(my_module,                 # model being run
                  (dummy_A, dummy_B),        # model input (or a tuple for multiple inputs)
                  "AplusB.onnx",             # where to save the model (can be a file or file-like object)
                  verbose=True,              # Print informations
                  export_params=True,        # store the trained parameter weights inside the model file
                  # opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=input_names,   # the model's input names
                  output_names=output_names, # the model's output names
                  dynamic_axes=dynamic_axes) # the model's dynamic axes



# dummy_A = torch.Tensor(3)
# dummy_B = torch.Tensor(3)

# dummy_A = dummy_A[None, :]
# dummy_B = dummy_B[None, :]                  

# torch.onnx.export(my_module, (example_A, example_B), 
#                   "AplusB.onnx", verbose=True, 
#                   input_names=input_names, output_names=output_names)
# torch.onnx.export(sm, "testwxb.onnx")