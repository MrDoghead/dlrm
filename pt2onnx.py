import torch
import onnx

# inputs: 
# {'input.1': {'shape': [2048, 13], 'dtype': <class 'numpy.float32'>}, 
# 'lS_o': {'shape': [26, 2048], 'dtype': <class 'numpy.int64'>}, 
# 'lS_i': {'shape': [26, 2048], 'dtype': <class 'numpy.int64'>}}
# outputs:
# [Variable (299): (shape=[2048, 1], dtype=float32)]

def export2onnx(pt_path,onnx_path):
    bs = 2
    device = torch.device('cpu')
    pt_model = torch.load(pt_path)
    dense_x = torch.randn(size=(bs,13), device=device)
    lS_o = torch.randint(low=0, high=1, size=(26,bs), device=device, dtype=torch.int64)
    lS_i = torch.randint(low=0, high=20000, size=(26,bs), device=device, dtype=torch.int64)

    dummy_inputs = {
            "dense_x": dense_x,
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
                opset_version=10,
                verbose=True,
                input_names=list(dummy_inputs.keys()),
                use_external_data_format=True,
                )

    # validate
    #print('validate onnx')
    #onnx_model = onnx.load(onnx_path)
    #onnx.checker.check_model(onnx_model)
    #print(f'ONNX is good and saved at {onnx_path}')

if __name__=="__main__":
    pt_path = "/home/ubuntu/project/mlcommons/inference/recommendation/dlrm/pytorch/tmp_dlrm.pt"
    onnx_path="./fake_tb0875_10M/dlrm_s_pytorch.onnx"
    export2onnx(pt_path,onnx_path)
