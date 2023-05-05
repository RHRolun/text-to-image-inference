import torch
import requests
import numpy as np

def inference_vae_decoder(data):
    endpoint = "https://vae-decoder-robert-serving-test.apps.rhods-internal.61tk.p1.openshiftapps.com/v2/models/vae-decoder/infer"
    data = torch.zeros((1, 4, 64, 64)).tolist()
    payload = {
        "inputs": [
            {
                "name": "latents", 
                "shape": np.shape(data),#[1, 4, 64, 64], 
                "datatype": "FP32",
                "data": data
            },
            ]
        }
    headers = {
        'content-type': 'application/json'
    }

    response = requests.post(endpoint, json=payload, headers=headers)
    import pdb; pdb.set_trace()
    prediction = response.json()['outputs'][0]['data']
    return prediction

data = torch.zeros((1, 4, 64, 64)).tolist()
inference = inference_vae_decoder(data)
import pdb; pdb.set_trace()

