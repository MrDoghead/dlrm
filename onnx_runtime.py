import onnx
import onnxruntime
import numpy as np
import onnx_graphsurgeon as gs
import pickle as pkl

def run(model_path, ort_data_path=None):
    print("=== onnxruntime session ===")
    onnxruntime.set_default_logger_severity(3)
    session = onnxruntime.InferenceSession(model_path)

    if ort_data_path:
        with open(ort_data_path, 'rb') as f:
            ort_data = pkl.load(f)
        inputs = ort_data['inputs']
        outputs = ort_data['outputs']
    else:
        print("generating inputs from onnx graph")
        gs_graph = gs.import_onnx(onnx.load(model_path))
        org_inputs = {inp.name: {'shape': inp.shape, 'dtype': inp.dtype.type} for inp in gs_graph.inputs}
        inputs = dict()
        n_table, bs = org_inputs['offsets']['shape']
        for inp in session.get_inputs():
            if inp.name == 'offsets':
                inputs[inp.name] = np.array([list(range(bs))] * n_table)
            else:
                inputs[inp.name] = np.random.random(size=inp.shape).astype(org_inputs[inp.name]['dtype'])
        outputs = [x.name for x in session.get_outputs()]

    print('inputs:',{k:v for k,v in inputs.items()})
    print('outputs names:', outputs)
    ort_outputs = session.run(outputs, inputs)
    print("output:",ort_outputs)

if __name__=="__main__":
    model_path = "./tmp_model/dlrm_s_pytorch.onnx"
    ort_data_path = './ort_data.pkl'
    run(model_path, ort_data_path=ort_data_path)
