import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from src.dentex_dataset import DentexCocoDataset, collate_fn
from train import build_model

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--images",required=True); p.add_argument("--annotations",required=True)
    p.add_argument("--checkpoint",required=True); p.add_argument("--output",default="evaluation_metrics.json")
    p.add_argument("--score-threshold",type=float,default=0.05); p.add_argument("--batch-size",type=int,default=2)
    args=p.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ds=DentexCocoDataset(args.images,args.annotations,transforms=T.Compose([T.ToImage(),T.ToDtype(torch.float32,scale=True)]))
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=2,collate_fn=collate_fn)
    ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False)
    model=build_model(ckpt["num_classes"]).to(device); model.load_state_dict(ckpt["model"]); model.eval()
    label_to_cat={int(k):v for k,v in ckpt["categories"].items()} if isinstance(next(iter(ckpt["categories"].keys())),str) else ckpt["categories"]
    results=[]
    with torch.inference_mode():
        for images,targets in loader:
            preds=model([x.to(device) for x in images])
            for pred,target in zip(preds,targets):
                image_id=int(target["image_id"])
                for box,label,score in zip(pred["boxes"].cpu(),pred["labels"].cpu(),pred["scores"].cpu()):
                    s=float(score)
                    if s<args.score_threshold: continue
                    x1,y1,x2,y2=map(float,box.tolist())
                    cat_id=int(label_to_cat[int(label)]["id"])
                    results.append({"image_id":image_id,"category_id":cat_id,"bbox":[x1,y1,x2-x1,y2-y1],"score":s})
    gt=COCO(args.annotations)
    if not results: raise RuntimeError("No predictions above threshold")
    dt=gt.loadRes(results)
    ev=COCOeval(gt,dt,"bbox"); ev.evaluate(); ev.accumulate(); ev.summarize()
    names={c["id"]:c["name"] for c in gt.dataset["categories"]}
    per_class={}
    for cid,name in names.items():
        e=COCOeval(gt,dt,"bbox"); e.params.catIds=[cid]; e.evaluate(); e.accumulate()
        precision=e.eval["precision"]
        valid=precision[precision>-1]
        per_class[name]={"mAP_50_95":float(valid.mean()) if valid.size else None}
    metrics={
        "checkpoint_epoch":int(ckpt.get("epoch",0)),
        "validation_images":len(ds),
        "predictions":len(results),
        "mAP_50_95":float(ev.stats[0]),"mAP_50":float(ev.stats[1]),"mAP_75":float(ev.stats[2]),
        "AR_100":float(ev.stats[8]),"per_class":per_class,
        "note":"Image-level held-out validation split; not external clinical validation."
    }
    Path(args.output).write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
