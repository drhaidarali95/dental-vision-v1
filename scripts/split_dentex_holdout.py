import argparse, json, random
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--annotations",required=True)
    p.add_argument("--train-out",required=True)
    p.add_argument("--val-out",required=True)
    p.add_argument("--val-fraction",type=float,default=0.2)
    p.add_argument("--seed",type=int,default=20260920)
    args=p.parse_args()
    coco=json.loads(Path(args.annotations).read_text())
    ids=[im["id"] for im in coco["images"]]
    rng=random.Random(args.seed); rng.shuffle(ids)
    n_val=max(1,round(len(ids)*args.val_fraction))
    val_ids=set(ids[:n_val]); train_ids=set(ids[n_val:])
    def subset(keep):
        return {
            **{k:v for k,v in coco.items() if k not in ("images","annotations")},
            "images":[x for x in coco["images"] if x["id"] in keep],
            "annotations":[a for a in coco["annotations"] if a["image_id"] in keep],
        }
    train,val=subset(train_ids),subset(val_ids)
    Path(args.train_out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.val_out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.train_out).write_text(json.dumps(train))
    Path(args.val_out).write_text(json.dumps(val))
    print(f"seed={args.seed} train_images={len(train['images'])} val_images={len(val['images'])}")
    print(f"train_annotations={len(train['annotations'])} val_annotations={len(val['annotations'])}")
if __name__=="__main__": main()
