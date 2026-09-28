#!/usr/bin/env python3
"""Run and rank AFPO experiments over deterministic CSV folds."""
import argparse, json, re, subprocess, sys, tempfile, time
from pathlib import Path
import pandas as pd
from afpo import kfold_split_indices

ROOT=Path(__file__).resolve().parent
VALIDATION_METRICS=re.compile(r"Validation \((?:used|not used) for selection\): loss=([^,\s]+), shape=([^,\s]+)")

def parse_validation_metrics(output):
 matches=VALIDATION_METRICS.findall(output)
 if not matches: return None
 loss,shape=matches[-1]
 return {"loss":float(loss),"shape":float(shape)}

def experiment_key(result):
 names=("operators","population","nodes","depth","bayesian_proposal_rate","crossover_rate")
 return tuple(result["configuration"][name] for name in names)

def summarize_experiments(results):
 """Aggregate completed fold/repeat measurements and rank lower loss first."""
 groups={}
 for result in results:
  metrics=result.get("validation")
  if result["returncode"] or metrics is None: continue
  groups.setdefault(experiment_key(result),[]).append(metrics)
 names=("operators","population","nodes","depth","bayesian_proposal_rate","crossover_rate")
 summary=[]
 for key,measurements in groups.items():
  summary.append({"configuration":dict(zip(names,key)),"completed_runs":len(measurements),
                  "mean_validation_loss":sum(m["loss"] for m in measurements)/len(measurements),
                  "mean_validation_shape":sum(m["shape"] for m in measurements)/len(measurements)})
 return sorted(summary,key=lambda item:(item["mean_validation_loss"],item["mean_validation_shape"]))

def main():
 p=argparse.ArgumentParser()
 p.add_argument("dataset"); p.add_argument("--folds",type=int,default=5)
 p.add_argument("--column-types",required=True,help="Comma-separated answers, e.g. '1*3,0*2,1,5*2'")
 p.add_argument("--operators",default="1,2,3,5",help="Operator-menu answer; use --operator-sets for a search")
 p.add_argument("--operator-sets",help="Semicolon-separated operator-menu answers to compare")
 p.add_argument("--generations",type=int,default=50); p.add_argument("--population",type=int,default=48)
 p.add_argument("--populations",help="Comma-separated population sizes to compare")
 p.add_argument("--nodes",type=int,default=31); p.add_argument("--node-limits",help="Comma-separated node limits to compare")
 p.add_argument("--depth",type=int,default=6); p.add_argument("--depths",help="Comma-separated depth limits to compare")
 p.add_argument("--bayesian-proposal-rate",type=float,default=.25); p.add_argument("--bayesian-proposal-rates",help="Comma-separated proposal rates to compare")
 p.add_argument("--crossover-rate",type=float,default=.35); p.add_argument("--crossover-rates",help="Comma-separated crossover rates to compare")
 p.add_argument("--repeats",type=int,default=1,help="Independent seeds per fold/configuration")
 p.add_argument("--seed",type=int,default=1)
 p.add_argument("--delimiter",default=",")
 p.add_argument("--output",type=Path,default=ROOT/"cross_validation_results.json",help="Where to write run-level and ranked results")
 a=p.parse_args(); dataset=Path(a.dataset); df=pd.read_csv(dataset,sep=a.delimiter)
 if a.repeats < 1: p.error("--repeats must be at least 1")
 def choices(value, multiple, cast=str): return [cast(x) for x in multiple.split(",")] if multiple else [value]
 configurations=[{"operators":operators,"population":population,"nodes":nodes,"depth":depth,
                  "bayesian_proposal_rate":rate,"crossover_rate":crossover}
                 for operators in (a.operator_sets.split(";") if a.operator_sets else [a.operators])
                 for population in choices(a.population,a.populations,int)
                 for nodes in choices(a.nodes,a.node_limits,int)
                 for depth in choices(a.depth,a.depths,int)
                 for rate in choices(a.bayesian_proposal_rate,a.bayesian_proposal_rates,float)
                 for crossover in choices(a.crossover_rate,a.crossover_rates,float)]
 folds=kfold_split_indices(len(df),a.folds,a.seed); results=[]
 with tempfile.TemporaryDirectory(prefix="afpo_cv_") as directory:
  temp=Path(directory)
  for i,valid in enumerate(folds):
   train=df.drop(index=valid); validation=df.iloc[valid]
   train_path=temp/f"fold{i}_train.csv"; valid_path=temp/f"fold{i}_validation.csv"
   train.to_csv(train_path,index=False); validation.to_csv(valid_path,index=False)
   for configuration in configurations:
    for repeat in range(a.repeats):
     seed=a.seed+i+repeat*a.folds
     answers=["0",str(train_path),"0",*a.column_types.split(","),configuration["operators"],"0","1","0",str(configuration["nodes"]),str(configuration["depth"]),str(valid_path),"1"]
     command=[sys.executable,"afpo.py","--max-generations",str(a.generations),"--population",str(configuration["population"]),"--seed",str(seed),"--bayesian-proposal-rate",str(configuration["bayesian_proposal_rate"]),"--crossover-rate",str(configuration["crossover_rate"])]
     started=time.time(); run=subprocess.run(command,cwd=ROOT,input="\n".join(answers)+"\n",text=True,capture_output=True)
     metrics=parse_validation_metrics(run.stdout) if run.returncode==0 else None
     results.append({"fold":i,"repeat":repeat,"seed":seed,"configuration":configuration,"rows_train":len(train),"rows_validation":len(validation),"returncode":run.returncode,"seconds":round(time.time()-started,3),"validation":metrics,"tail":run.stdout.splitlines()[-12:]})
     print(f"fold {i+1}/{a.folds}, repeat {repeat+1}/{a.repeats}: {'PASS' if metrics is not None else 'FAIL'}")
 summary=summarize_experiments(results)
 a.output.write_text(json.dumps({"runs":results,"ranked_configurations":summary},indent=2)+"\n")
 if summary: print("Best configuration:",json.dumps(summary[0],sort_keys=True))
 if any(x["returncode"] or x["validation"] is None for x in results): raise SystemExit(1)
if __name__=="__main__": main()
