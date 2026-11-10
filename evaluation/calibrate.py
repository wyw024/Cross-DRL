"""Reproduce traffic-scale and ego-car smoke checks; no model training."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from environment.sumo_env import IntersectionEnv, SUMO_HOME

records=[]
with tempfile.TemporaryDirectory(prefix='sumo_calibrate_') as temporary:
    for density,scale in IntersectionEnv.DENSITY_SCALE.items():
        summary=Path(temporary)/f'{density}.xml'
        cmd=[str(SUMO_HOME/'bin'/'sumo.exe'),'-c',str(ROOT/'environment'/'network'/'intersection.sumocfg'),
             '--scale',str(scale),'--seed','42','--no-step-log','true','--summary-output',str(summary)]
        subprocess.run(cmd,check=True)
        rows=ET.parse(summary).getroot().findall('step')
        last=rows[-1]
        records.append({'kind':'background','density':density,'seed':42,'end_time_s':float(last.get('time')),
                        'inserted':int(last.get('inserted')),'ended':int(last.get('ended')),
                        'running_at_end':int(last.get('running')),
                        'peak_halting':max(int(r.get('halting')) for r in rows),
                        'mean_halting':sum(int(r.get('halting')) for r in rows)/len(rows)})
        for mode in ('natural','controlled_red'):
            env=IntersectionEnv(traffic_density=density,max_steps=300)
            try:
                _,info=env.reset(seed=42)
                if mode=='controlled_red':
                    c=env.connection
                    tls,index,_,_=c.vehicle.getNextTLS('ego')[0]
                    phases=c.trafficlight.getAllProgramLogics(tls)[0].phases
                    red=next(i for i,p in enumerate(phases) if p.state[index].lower()=='r')
                    c.trafficlight.setPhase(tls,red)
                    c.trafficlight.setPhaseDuration(tls,40)
                observations=[]
                for step in range(300):
                    _,_,terminated,truncated,info=env.step(4)
                    observations.append(info)
                    if terminated or truncated: break
                records.append({'kind':'ego','mode':mode,'density':density,'seed':42,'steps':step+1,
                    'passed':info['passed_intersection'],'collision':info['collision'],
                    'red_violation':info['red_light_violation'],
                    'zero_speed_steps':sum(x.get('speed_m_s',1)<.1 for x in observations),
                    'red_signal_steps':sum(str(x.get('signal','')).lower()=='r' for x in observations),
                    'leader_observed_steps':sum(x.get('leader_gap_m') is not None for x in observations),
                    'minimum_reported_leader_gap_m':min((x['leader_gap_m'] for x in observations if x.get('leader_gap_m') is not None),default=None)})
            finally: env.close()

out=ROOT/'results'/'logs'/'calibration.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(records,ensure_ascii=False,indent=2))
print('Saved',out)
