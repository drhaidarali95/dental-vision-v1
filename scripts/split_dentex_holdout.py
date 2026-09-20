import argparse, json, random
from collections import Counter
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
    cats=[c["id"] for c in coco.get("categories",[])]
    by_img={i:set() for i in ids}
    for a in coco.get("annotations",[]):
        if a["image_id"] in by_img: by_img[a["image_id"]].add(a["category_id"])

    # Greedy multilabel-stratified holdout: preserve rare disease classes first.
    rng=random.Random(args.seed)
    target_n=max(1,round(len(ids)*args.val_fraction))
    total=Counter(c for s in by_img.values() for c in s)
    target={c:max(1,round(total[c]*args.val_fraction)) for c in cats}
    remaining=ids[:]; rng.shuffle(remaining); val=[]; counts=Counter()
    while remaining and len(val)<target_n:
        def gain(i):
            return sum(max(target[c]-counts[c],0)/max(target[c],1) for c in by_img[i])
        best=max(remaining,key=lambda i:(gain(i),rng.random()))
        val.append(best); remaining.remove(best)
        for c in by_img[best]: counts[c]+=1
    val_ids=set(val); train_ids=set(ids)-val_ids

    def subset(keep):
        return {**{k:v for k,v in coco.items() if k not in ("images","annotations")},
                "images":[x for x in coco["images"] if x["id"] in keep],
                "annotations":[a for a in coco["annotations"] if a["image_id"] in keep]}
    train,val=subset(train_ids),subset(val_ids)
    Path(args.train_out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.val_out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.train_out).write_text(json.dumps(train)); Path(args.val_out).write_text(json.dumps(val))
    def dist(x): return Counter(a["category_id"] for a in x["annotations"])
    print(f"seed={args.seed} train_images={len(train['images'])} val_images={len(val['images'])}")
    print("train_class_annotations=",dict(dist(train)))
    print("val_class_annotations=",dict(dist(val)))
    assert all(dist(val)[c]>0 for c in cats), "A disease class is absent from validation split"
if __name__=="__main__": main()
