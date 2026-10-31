"""Fixed-recipe training helpers extracted unchanged from the frozen search code."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import torch
from torch import nn
import lightgbm as lgb
from training import probability
from gpu_residual_submission import network_input,forward
DEVICE=torch.device('cuda:0')
ROUNDS=(20,60,160)
torch.set_num_threads(4);torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.use_deterministic_algorithms(True)

class Residual(nn.Module):
    def __init__(self,d,width,drop,bound):
        super().__init__();self.enc=nn.Linear(d,width);self.drop=nn.Dropout(drop);self.out=nn.Linear(width,1);self.bound=bound
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,x):return self.bound*torch.tanh(self.out(self.drop(torch.relu(self.enc(x))))[...,0])

def fit_network(manual,anchor,y,valid,rows,held,spec,seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    raw=network_input(manual,anchor);mu=raw[rows].mean((0,1));sd=raw[rows].std((0,1)).clip(.01)
    z=np.clip((raw-mu)/sd,-8,8).astype(np.float32)
    xt=torch.from_numpy(z).to(DEVICE);at=torch.from_numpy(anchor.astype(np.float32)).to(DEVICE)
    yt=torch.from_numpy(y.astype(np.float32)).to(DEVICE);vt=torch.from_numpy(valid).to(DEVICE)
    index=torch.from_numpy(np.flatnonzero(rows)).to(DEVICE)
    held_index=torch.from_numpy(np.flatnonzero(held) if held.dtype==bool else held).to(DEVICE)
    model=Residual(raw.shape[-1],spec['width'],spec['drop'],spec['bound']).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05)
    snaps={};states={}
    for epoch in range(1,61):
        model.train()
        for batch in index[torch.randperm(len(index),device=DEVICE)].split(128):
            residual=model(xt[batch]);score=(at[batch]+residual).masked_fill(~vt[batch],-1e9)
            target=yt[batch]/yt[batch].sum(1,keepdim=True)
            loss=-(target*torch.log_softmax(score,1)).sum(1).mean()+.02*residual.square().mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if epoch in (10,30,60):
            model.eval()
            with torch.no_grad():cuda_out=model(xt[held_index]).cpu().numpy()
            state={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()};state.update(mu=mu,sd=sd)
            cpu_out=forward(raw[held],state,spec['bound'])
            np.testing.assert_allclose(cpu_out,cuda_out,atol=3e-4,rtol=3e-4)
            snaps[epoch]=cpu_out;states[epoch]=state
    return snaps,states

def export(model,iterations):
    feature=[];threshold=[];left=[];right=[];value=[];roots=[];depths=[]
    def visit(node,depth):
        index=len(feature);feature.append(-1);threshold.append(0.);left.append(-1);right.append(-1)
        value.append(node.get('leaf_value',0.));depths.append(depth)
        if 'split_feature' in node:
            assert node['decision_type']=='<=' and not node.get('missing_type')=='NaN'
            feature[index]=node['split_feature'];threshold[index]=node['threshold']
            left[index]=visit(node['left_child'],depth+1);right[index]=visit(node['right_child'],depth+1)
        return index
    for tree in model.dump_model(num_iteration=iterations)['tree_info']:roots.append(visit(tree['tree_structure'],0))
    return dict(feature=np.asarray(feature,np.int32),threshold=np.asarray(threshold,np.float64),left=np.asarray(left,np.int32),
        right=np.asarray(right,np.int32),value=np.asarray(value,np.float64),roots=np.asarray(roots,np.int32),
        max_depth=np.array(max(depths,default=0),np.int32))

def fit_tree(x,anchor,y,valid,rows,spec):
    yy=y[rows].astype(np.float64);vv=valid[rows];target=yy/yy.sum(1,keepdims=True)
    first=[True]
    def objective(predictions,dataset):
        score=predictions.reshape(-1,4)
        if first[0]:
            np.testing.assert_allclose(score,anchor[rows],atol=1e-12,rtol=1e-12);first[0]=False
        if spec['loss']=='ce':
            p=probability(score,vv);gradient=p-target;hessian=np.maximum(p*(1-p),1e-6)
        else:
            pairs=yy[:,:,None]*(1-yy[:,None,:])*vv[:,None,:];pairs/=np.maximum(pairs.sum((1,2),keepdims=True),1)
            diff=score[:,:,None]-score[:,None,:];q=1/(1+np.exp(np.clip(diff,-40,40)))
            z=pairs*q;gradient=-z.sum(2)+z.sum(1)
            z=pairs*q*(1-q);hessian=np.maximum(z.sum(2)+z.sum(1),1e-6)
        return gradient.ravel(),hessian.ravel()
    params=dict(objective=objective,num_leaves=spec['leaves'],learning_rate=.03,min_data_in_leaf=60,
        lambda_l2=10.,feature_fraction=.8,verbosity=-1,num_threads=4,seed=73,deterministic=True,force_col_wise=True,
        max_bin=63,min_sum_hessian_in_leaf=1.,boost_from_average=False)
    dataset=lgb.Dataset(x[rows].reshape(-1,x.shape[-1]),label=y[rows].ravel(),init_score=anchor[rows].ravel(),free_raw_data=True)
    return lgb.train(params,dataset,num_boost_round=max(ROUNDS))

def fit_seeded(x,anchor,y,valid,rows,spec,seed):
    target=y[rows]/y[rows].sum(1,keepdims=True);vv=valid[rows]
    first=[True]
    def objective(pred,dataset):
        score=pred.reshape(-1,4)
        if first[0]:np.testing.assert_allclose(score,anchor[rows],atol=1e-12,rtol=1e-12);first[0]=False
        p=probability(score,vv)
        return (p-target).ravel(),np.maximum(p*(1-p),1e-6).ravel()
    params=dict(objective=objective,num_leaves=spec['leaves'],min_data_in_leaf=spec['min_leaf'],lambda_l2=spec['l2'],
        learning_rate=.03,feature_fraction=.8,verbosity=-1,num_threads=4,seed=seed,deterministic=True,force_col_wise=True,
        max_bin=63,min_sum_hessian_in_leaf=1.,boost_from_average=False)
    dataset=lgb.Dataset(x[rows].reshape(-1,x.shape[-1]),label=y[rows].ravel(),init_score=anchor[rows].ravel(),free_raw_data=True)
    return lgb.train(params,dataset,num_boost_round=max(ROUNDS))
