import onnx
import onnxruntime
import numpy as np

model_path = "../tb0875_10M/dlrm_s_pytorch.onnx"
onnxruntime.set_default_logger_severity(3)
session = onnxruntime.InferenceSession(model_path)
print("session created")

# This file `./dlrm_inputs.npy` gives 10 critieo samples for testing onnxruntime
# Due to some issue in torch.export_onnx, currently only support batch_size=1
# You can specify `idx=0~9` to get an input data.
data = np.load('./dlrm_inputs.npz') 
idx = 0 
inputs = dict()
inputs['dense_x'] = data['dense_x'][idx,:].reshape(1,13)
inputs['offsets'] = data['offsets'][:,idx].reshape(26,1)
inputs['indices_0'] = np.array([data['indices_0'][idx]])
inputs['indices_1'] = np.array([data['indices_1'][idx]])
inputs['indices_2'] = np.array([data['indices_2'][idx]])
inputs['indices_3'] = np.array([data['indices_3'][idx]])
inputs['indices_4'] = np.array([data['indices_4'][idx]])
inputs['indices_5'] = np.array([data['indices_5'][idx]])
inputs['indices_6'] = np.array([data['indices_6'][idx]])
inputs['indices_7'] = np.array([data['indices_7'][idx]])
inputs['indices_8'] = np.array([data['indices_8'][idx]])
inputs['indices_9'] = np.array([data['indices_9'][idx]])
inputs['indices_10'] = np.array([data['indices_10'][idx]])
inputs['indices_11'] = np.array([data['indices_11'][idx]])
inputs['indices_12'] = np.array([data['indices_12'][idx]])
inputs['indices_13'] = np.array([data['indices_13'][idx]])
inputs['indices_14'] = np.array([data['indices_14'][idx]])
inputs['indices_15'] = np.array([data['indices_15'][idx]])
inputs['indices_16'] = np.array([data['indices_16'][idx]])
inputs['indices_17'] = np.array([data['indices_17'][idx]])
inputs['indices_18'] = np.array([data['indices_18'][idx]])
inputs['indices_19'] = np.array([data['indices_19'][idx]])
inputs['indices_20'] = np.array([data['indices_20'][idx]])
inputs['indices_21'] = np.array([data['indices_21'][idx]])
inputs['indices_22'] = np.array([data['indices_22'][idx]])
inputs['indices_23'] = np.array([data['indices_23'][idx]])
inputs['indices_24'] = np.array([data['indices_24'][idx]])
inputs['indices_25'] = np.array([data['indices_25'][idx]])
outputs = ['pred']
print('inputs:', {k:v for k,v in inputs.items()})
print('outputs names:', outputs)

# onnx runtime
ort_outputs = session.run(outputs, inputs)
print('pred:',ort_outputs)
