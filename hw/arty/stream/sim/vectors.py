"""Synthetic three-layer fixtures, exact RN32 dot products and high-precision tanh.
This creates development data, not a checkpoint export or RL final evaluation.
"""
import csv
import importlib.util
import json
from pathlib import Path
import random


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def generate(out,arty,seed):
    fp=module('stream_fp_reference',arty/'fp32/sim/reference.py')
    tanh=module('stream_tanh_reference',arty/'tanh/sim/model.py')
    jobs=[]
    for job,actor in enumerate([0,1,0,1]):
        rng=random.Random(seed+job*101);dims=[19 if actor else 25,256,256,2]
        inputs=[fp.bits(rng.uniform(-.9,.9)) for _ in range(dims[0])]
        # Finite signed zeros/subnormals exercise the stream without violating
        # the finite-input job contract. First pair also carries an FMA canary.
        inputs[0:4]=[0x3f7fffff,0x80000000,1,0x80000001]
        bank=inputs;jobflags=0;weights=[];biases=[];layers=[]
        for layer,(ni,no) in enumerate(zip(dims,dims[1:])):
            w=[[fp.bits(rng.uniform(-.125,.125)) for _ in range(ni)] for _ in range(no)]
            b=[fp.bits(rng.uniform(-.1,.1)) for _ in range(no)]
            if layer==0:
                w[0]=[0x3f800001]+[0]*(ni-1);b[0]=0xbf800000
                # Negative constant is clipped; a positive bias checks bias-first.
                w[1]=[0]*ni;b[1]=0xbf000000
                w[2]=[0]*ni;b[2]=0x3e800000
            pre=[];post=[]
            for row in range(no):
                total=b[row]
                for col in range(ni):
                    product,mf=fp.operate(w[row][col],bank[col],True)
                    total,af=fp.operate(total,product);jobflags|=mf|af
                pre.append(total)
                if layer<2:
                    post.append(0 if (total>>31 and total&0x7fffffff) else total)
                else:
                    y,invalid,_,_=tanh.evaluate(total);gold,gi=tanh.reference(total,180)
                    assert not invalid and not gi and abs(y-gold)<=1
                    post.append(y)
            if layer==0:assert post[:3]==[0,0,0x3e800000]
            layers.append({'input':bank,'bias':b,'weights':w,'pre':pre,'post':post})
            weights += [w[row][col] for col in range(ni) for row in range(no)]
            biases += b;bank=post
        prefix=f'job{job}'
        for name,values in [('input',inputs),('weight',weights),('bias',biases),
                            ('pre',[v for l in layers for v in l['pre']]),
                            ('post',[v for l in layers for v in l['post']]),('output',bank)]:
            (out/f'{prefix}-{name}.hex').write_text(''.join(f'{v:08x}\n' for v in values))
        (out/f'{prefix}-flags.hex').write_text(f'{jobflags:02x}\n')
        with (out/f'{prefix}-expected.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['layer','row','pre_activation','post_activation'])
            for i,l in enumerate(layers):
                for r,(pre,post) in enumerate(zip(l['pre'],l['post'])):writer.writerow([i,r,f'{pre:08x}',f'{post:08x}'])
        jobs.append({'index':job,'actor':actor,'dims':dims,'seed':seed+job*101,'scalar_macs':len(weights),
                     'weights':len(weights),'biases':len(biases),'outputs':[f'{v:08x}' for v in bank],
                     'dense_flags':jobflags,'fixtures':prefix})
    (out/'fixtures.json').write_text(json.dumps(jobs,indent=2)+'\n');return jobs
