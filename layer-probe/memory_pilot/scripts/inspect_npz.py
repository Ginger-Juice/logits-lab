import numpy as np, json
P='/root/autodl-tmp/layer-probe/pipeline_v1/outputs/'
z=np.load(P+'features_off.npz', allow_pickle=False)
for k in z.files:
    a=z[k]; print(k, a.shape, a.dtype, a.ravel()[:5] if a.size<100 else '')
c=json.load(open('/root/autodl-tmp/qwen38-27b-fp8/config.json'))
tc=c.get('text_config',c)
lt=tc.get('layer_types'); print('nlayers',tc.get('num_hidden_layers'), 'hidden', tc.get('hidden_size'))
print('full idx (1-based):',[i+1 for i,t in enumerate(lt) if t!='linear_attention'], set(lt))
print(open(P+'capture_meta.json').read()[:2742])
r=json.loads(open(P+'generations.jsonl').readline()); print({k:(v if not isinstance(v,(list,str)) or len(v)<80 else (type(v).__name__,len(v))) for k,v in r.items()})
