import onnx
import onnxruntime
import numpy as np
import onnx_graphsurgeon as gs

#model_path = "./tb0875_10M/dlrm_s_pytorch.onnx"
model_path = "./fake_tb0875_10M/dlrm_s_pytorch.onnx"
onnxruntime.set_default_logger_severity(3)
session = onnxruntime.InferenceSession(model_path)
print("session created")

gs_graph = gs.import_onnx(onnx.load(model_path))
org_inputs = {inp.name: {'shape': inp.shape, 'dtype': inp.dtype.type} for inp in gs_graph.inputs}
inputs = dict()
for inp in session.get_inputs():
    inputs[inp.name] = np.random.random(size=inp.shape).astype(org_inputs[inp.name]['dtype'])
outputs = [x.name for x in session.get_outputs()]

print('inputs:',{k:v.shape for k,v in inputs.items()})
print('outputs names:', outputs)
ort_outputs = session.run(outputs, inputs)
print(ort_outputs)

