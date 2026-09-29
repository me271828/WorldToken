"""Leading analytic temporal FLOPs and optional exact trainable parameter counts."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common.configuration import configs, read_config
from experiments.common.results import write_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts",type=int,nargs="+",default=[1,10,50,100,250,500])
    parser.add_argument("--output-dir",type=Path,default=Path("results/05_token_interface"))
    parser.add_argument("--count-parameters",action="store_true",
                        help="Instantiate each N2 model on CPU; requires the model dependencies")
    args=parser.parse_args()
    if any(c<1 for c in args.contexts): parser.error("Contexts must be positive")
    selected={}
    for path in configs():
        cfg=read_config(path);p=cfg["paper"]
        if p.get("D")==300 and p.get("N")==2 and p.get("seed")==0 and p.get("C_train")==10:
            selected[p["K"]]=cfg
    rows=[];counts=[]
    for k in (1,4,50):
        cfg=selected[k];temporal=cfg["sequence_model"];params=temporal["params"]
        width=temporal["hidden_dim"];ffn=params["ffn_hidden_size"];layers=params["n_layers"]
        linear=layers*(8*width*width+6*width*ffn)
        attention=4*layers*width
        for c in args.contexts:
            n=c*k;full=linear*n+attention*n*n;reference=linear*c+attention*c*c
            rows.append(dict(K=k,C=c,temporal_tokens=n,full_prefix_flops=full,
                             full_prefix_ratio_to_k1=full/reference,
                             analytic_cached_query_flops=linear*k+attention*k*n))
        if args.count_parameters:
            from worldtoken.builder import build_model
            from worldtoken.config import load_config
            model,_=build_model(load_config(cfg),device="cpu")
            counts.append(dict(K=k,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
            del model
    write_table(args.output_dir,"temporal_compute",rows)
    if counts: write_table(args.output_dir,"parameter_counts",counts)


if __name__ == "__main__":
    main()
