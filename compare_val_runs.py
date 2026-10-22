"""Validation-only comparison of runs from saved best_val_predictions.csv (+ history.csv). No GPU, no test data.
Usage: python scripts\compare_val_runs.py NAME=runs\dirA NAME=runs\dirB ... [--out val_comparison.json]
Exploratory: one seed per arm gives a direction, not significance."""
import sys, json, csv
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, cohen_kappa_score, f1_score
L=['MA','HE','IH','VB_IRMA','NV','VH','RD']; E=['VB_IRMA','NV','VH','RD']; R=['NV','VH','RD']
def js(x,y):
    x=np.clip(x,1e-6,1-1e-6); y=np.clip(y,1e-6,1-1e-6); m=(x+y)/2
    kl=lambda z:z*(np.log(z)-np.log(m))+(1-z)*(np.log1p(-z)-np.log1p(-m))
    return (kl(x)+kl(y))/2
def analyse(d):
    d=Path(d); rows=list(csv.DictReader((d/'best_val_predictions.csv').open(encoding='utf-8')))
    y=np.array([int(r['grade']) for r in rows]); pred=np.array([int(r['prediction']) for r in rows])
    g=np.array([[float(r[f'g{i}']) for i in range(5)] for r in rows])
    t=np.array([[float(r['target_'+c]) for c in L] for r in rows]); p=np.array([[float(r['p_'+c]) for c in L] for r in rows])
    ap={c:float(average_precision_score(t[:,i],p[:,i])) for i,c in enumerate(L)}
    out={'dir':str(d),'n':len(y),'qwk':float(cohen_kappa_score(y,pred,weights='quadratic')),
         'macro_f1':float(f1_score(y,pred,average='macro')),'ap':ap,'rare_ap_E':float(np.mean([ap[c] for c in E])),
         'rare_ap_R':float(np.mean([ap[c] for c in R]))}
    h=d/'history.csv'
    if h.exists():
        hr=list(csv.DictReader(h.open())); b=max(hr,key=lambda r:float(r['selection_score']))
        out['best_epoch']=int(b['epoch']); out['selection_score']=float(b['selection_score']); out['epochs_in_history']=len(hr)
        out['raw_consistency_loss_ep1_20_last']=[float(hr[i]['train_consistency_loss']) for i in (0,min(19,len(hr)-1),len(hr)-1)]
    a=1-np.prod(1-p,axis=1); bb=1-np.prod(1-p[:,4:7],axis=1)
    anyg=1-g[:,0]; advg=g[:,4]
    out['disagreement_continuous_mean']={'any_lesion_vs_grade>0':float(js(a,anyg).mean()),'advanced_vs_grade4':float(js(bb,advg).mean())}
    for th in (0.3,0.5,0.7):  # threshold stated explicitly; report all so no single threshold drives conclusions
        out[f'disagreement_binary_th{th}']={'any_lesion_vs_grade>0':float(((a>=th)!=(anyg>=th)).mean()),
                                            'advanced_vs_grade4':float(((bb>=th)!=(advg>=th)).mean())}
    out['lesion_prob_pos_median']={c:float(np.median(p[t[:,i]==1,i])) for i,c in enumerate(L)}
    out['lesion_prob_neg_median']={c:float(np.median(p[t[:,i]==0,i])) for i,c in enumerate(L)}
    out['false_negative_rate_at_0.5']={c:float((p[t[:,i]==1,i]<0.5).mean()) for i,c in enumerate(L)}
    out['false_positive_rate_at_0.5']={c:float((p[t[:,i]==0,i]>=0.5).mean()) for i,c in enumerate(L)}
    return out
if __name__=='__main__':
    args=[a for a in sys.argv[1:] if '=' in a]; outp=sys.argv[sys.argv.index('--out')+1] if '--out' in sys.argv else 'val_comparison.json'
    res={a.split('=',1)[0]:analyse(a.split('=',1)[1]) for a in args}
    Path(outp).write_text(json.dumps(res,indent=2),encoding='utf-8')
    names=list(res); print('metric'.ljust(34)+''.join(n.rjust(14) for n in names))
    def row(lbl,f): print(lbl.ljust(34)+''.join(f"{f(res[n]):14.4f}" for n in names))
    row('QWK',lambda r:r['qwk']); row('macro F1',lambda r:r['macro_f1']); row('rare AP (E)',lambda r:r['rare_ap_E']); row('rare AP (R)',lambda r:r['rare_ap_R'])
    for c in L: row('AP '+c,lambda r,c=c:r['ap'][c])
    row('JS any-lesion vs grade>0',lambda r:r['disagreement_continuous_mean']['any_lesion_vs_grade>0'])
    row('JS advanced vs grade4',lambda r:r['disagreement_continuous_mean']['advanced_vs_grade4'])
    for c in R+['VB_IRMA']: row('FN rate@0.5 '+c,lambda r,c=c:r['false_negative_rate_at_0.5'][c])
    print('saved',outp)
