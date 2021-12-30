import torch
import onnx

# inputs: 
# {'input.1': {'shape': [2048, 13], 'dtype': <class 'numpy.float32'>}, 
# 'lS_o': {'shape': [26, 2048], 'dtype': <class 'numpy.int64'>}, 
# 'lS_i': {'shape': [26, 2048], 'dtype': <class 'numpy.int64'>}}
# outputs:
# [Variable (299): (shape=[2048, 1], dtype=float32)]

def export2onnx(pt_model):
    device = torch.device('cpu')
    onnx_path="./tb0875_10M/dlrm_s_pytorch_10GB.onnx"
    input_1 = torch.randn(size=(2048,13), device=device)
    lS_o = torch.randint(low=0, high=1, size=(26,2048), device=device, dtype=torch.int64)
    lS_i = torch.randint(low=0, high=20000, size=(26,2048), device=device, dtype=torch.int64)

    dummy_inputs = {
            "input.1": input_1,
            "lS_o": lS_o,
            "lS_i": lS_i,
            }

    print("===== ONNX INFO =====")
    print("pt model:",type(pt_model))
    print("inputs names:",list(dummy_inputs.keys()))
    print("dummy inputs:",tuple(dummy_inputs.values()))
    torch.onnx.export(
                pt_model,
                tuple(dummy_inputs.values()),
                onnx_path,
                export_params=True,
                opset_version=11,
                verbose=True,
                input_names=list(dummy_inputs.keys()),
                use_external_data_format=True,
                )

    # validate
    print('validate onnx')
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f'ONNX is good and saved at {onnx_path}')
