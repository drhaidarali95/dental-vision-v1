import argparse, json
from collections import Counter
from pathlib import Path
from PIL import Image

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--images",required=True)
    p.add_argument("--annotations",required=True)
    p.add_argument("--output",default="geometry_audit.json")
    args=p.parse_args()
    d=json.loads(Path(args.annotations).read_text())
    ims={x["id"]:x for x in d["images"]}
    stats=Counter(); samples=[]
    for a in d["annotations"]:
        im=ims.get(a["image_id"])
        if not im: stats["missing_image_record"]+=1; continue
        fp=Path(args.images)/im["file_name"]
        if not fp.exists(): stats["missing_file"]+=1; continue
        with Image.open(fp) as pic: W,H=pic.size
        x,y,w,h=map(float,a["bbox"])
        stats["boxes"]+=1
        if x<0 or y<0 or w<=0 or h<=0: stats["invalid_basic"]+=1
        if x+w>W+1 or y+h>H+1: stats["outside_image"]+=1
        if max(x,y,w,h)<=1.5: stats["looks_normalized"]+=1
        if w>0.95*W and h>0.95*H: stats["near_full_image"]+=1
        if len(samples)<20:
            samples.append({"image":im["file_name"],"image_size":[W,H],"bbox":a["bbox"],"category_id":a["category_id"]})
    out={"images":len(ims),"annotations":len(d["annotations"]),"stats":dict(stats),"samples":samples}
    Path(args.output).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__": main()
