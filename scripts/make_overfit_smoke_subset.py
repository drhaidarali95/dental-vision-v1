import argparse, json, random
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--annotations",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--images",type=int,default=12)
    p.add_argument("--seed",type=int,default=20260920)
    args=p.parse_args()
    d=json.loads(Path(args.annotations).read_text())
    by={}
    for a in d["annotations"]: by.setdefault(a["image_id"],[]).append(a)
    ids=[i["id"] for i in d["images"] if by.get(i["id"])]
    random.Random(args.seed).shuffle(ids); keep=set(ids[:args.images])
    out={k:v for k,v in d.items() if k not in ("images","annotations")}
    out["images"]=[i for i in d["images"] if i["id"] in keep]
    out["annotations"]=[a for a in d["annotations"] if a["image_id"] in keep]
    Path(args.output).write_text(json.dumps(out))
    print(f"smoke images={len(out['images'])} annotations={len(out['annotations'])}")

if __name__=="__main__": main()
