#!/usr/bin/env python3
"""Isolated, simulation-only SO-101 paired asset benchmark (no training)."""
from __future__ import annotations
import argparse
import csv
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

import psutil

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
HOME = [0., -1.710422666954443, 1.605702911834783, 1.134464013796314, 0., -.087266462599716]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')
    temporary.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def gpu_sample():
    fields = ['utilization.gpu', 'memory.used', 'utilization.memory', 'power.draw', 'temperature.gpu']
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=' + ','.join(fields), '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
        values = r.stdout.strip().splitlines()[0].split(',')
        return {k: float(v.strip()) for k, v in zip(['gpu_percent','gpu_vram_mib','gpu_memory_controller_percent','gpu_power_w','gpu_temperature_c'], values)}
    except Exception as error:
        return {'gpu_sample_error': str(error)}


class Monitor:
    def __init__(self, path, roots):
        self.path, self.roots = path, roots
        self.stop_event = threading.Event()
        self.phase = 'startup'
        self.previous = {}
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        psutil.cpu_percent()
        with self.path.open('w', buffering=1) as stream:
            while not self.stop_event.is_set():
                started = time.monotonic()
                processes = {}
                for pid in list(self.roots):
                    try:
                        parent = psutil.Process(pid)
                        processes[parent.pid] = parent
                        processes.update({p.pid: p for p in parent.children(recursive=True)})
                    except psutil.Error:
                        pass
                cpu, rss, pss, missing = 0., 0, 0, 0
                details = []
                for pid, proc in processes.items():
                    try:
                        ct = proc.cpu_times()
                        total = ct.user + ct.system
                        key = (pid, proc.create_time())
                        prev = self.previous.get(key)
                        if prev and started > prev[0]:
                            cpu += max(0., total-prev[1])/(started-prev[0])*100
                        self.previous[key] = (started,total)
                        mem = proc.memory_full_info()
                        rss += mem.rss
                        if hasattr(mem,'pss'): pss += mem.pss
                        else: missing += 1
                        details.append({'pid':pid,'name':proc.name(),'rss_bytes':mem.rss,'pss_bytes':getattr(mem,'pss',None),'cpu_time_s':total})
                    except psutil.Error:
                        missing += 1
                vm = psutil.virtual_memory()
                record = dict(utc=timestamp(),monotonic_s=started,phase=self.phase,
                    cpu_process_percent=cpu,cpu_host_percent=psutil.cpu_percent(),
                    process_rss_mib=rss/2**20,process_pss_mib=pss/2**20,
                    process_count=len(details),memory_sample_missing_processes=missing,
                    system_available_mib=vm.available/2**20,system_used_mib=vm.used/2**20,
                    processes=details,**gpu_sample())
                stream.write(json.dumps(record)+'\n')
                self.stop_event.wait(max(0.,1.0-(time.monotonic()-started)))

    def close(self):
        self.stop_event.set(); self.thread.join(timeout=5)


def matching_processes(partitions):
    results=[]
    for proc in psutil.process_iter(['pid','name','cmdline','ppid']):
        try:
            env=proc.environ()
            if env.get('IGN_PARTITION') in partitions or env.get('GZ_PARTITION') in partitions:
                if proc.pid != os.getpid(): results.append(proc)
        except psutil.Error: pass
    return results


def clean_partitions(partitions):
    before=matching_processes(partitions)
    for proc in before:
        try:proc.terminate()
        except psutil.Error:pass
    _, alive=psutil.wait_procs(before,timeout=5)
    for proc in alive:
        try:proc.kill()
        except psutil.Error:pass
    psutil.wait_procs(alive,timeout=3)
    remaining=matching_processes(partitions)
    if remaining:raise RuntimeError(f'Incomplete cleanup: {[p.pid for p in remaining]}')
    return [p.pid for p in before]


def receive(connection, timeout=90):
    if not connection.poll(timeout):raise TimeoutError('Benchmark worker response timeout')
    result=connection.recv()
    if result.get('type')=='error':raise RuntimeError(json.dumps(result,ensure_ascii=False))
    return result


def command_all(connections, command, timeout=90):
    for c in connections:c.send(command)
    return [receive(c,timeout)for c in connections]


def gpu_environment(prefix, spec):
    env=os.environ.copy()
    for key in ('LIBGL_ALWAYS_SOFTWARE','GALLIUM_DRIVER','LP_NUM_THREADS','CUDA_VISIBLE_DEVICES'):
        env.pop(key,None)
    env.update(ROS_DOMAIN_ID=str(spec['domain_id']),IGN_PARTITION=spec['partition'],GZ_PARTITION=spec['partition'],
        ROS_LOCALHOST_ONLY='1',SO101_ENVIRONMENT_RANDOMIZATION_ENABLED='false',SO101_IMAGE_AUGMENTATION_ENABLED='false',
        ROS_LOG_DIR=str(Path(spec['output_dir'])/'ros_logs'),PYTHONNOUSERSITE='1',QT_QPA_PLATFORM='xcb')
    env['AMENT_PREFIX_PATH']=str(prefix)+os.pathsep+env.get('AMENT_PREFIX_PATH','')
    resources=os.pathsep.join([str(prefix/'share'),str(prefix/'share/so101_gazebo/models'),env.get('IGN_GAZEBO_RESOURCE_PATH','')])
    env['IGN_GAZEBO_RESOURCE_PATH']=resources
    env['GZ_SIM_RESOURCE_PATH']=resources
    plugins=os.pathsep.join(['/opt/ros/humble/lib',str(prefix/'lib'),env.get('IGN_GAZEBO_SYSTEM_PLUGIN_PATH','')])
    env['IGN_GAZEBO_SYSTEM_PLUGIN_PATH']=plugins
    env['GZ_SIM_SYSTEM_PLUGIN_PATH']=plugins
    env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
    return env


def spawn_log(command, env, path):
    with path.open('ab',buffering=0)as stream:
        return subprocess.Popen(command,env=env,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)


def child_entry(connection, spec, log_path):
    with open(log_path,'a',buffering=1)as f:
        os.dup2(f.fileno(),1);os.dup2(f.fileno(),2)
        from benchmark_worker import worker_main
        worker_main(connection,spec)


def fixed_motion(step):
    t=step/30.
    q=list(HOME)
    q[0]+=.10*math.sin(2*math.pi*t/8.)
    q[4]+=.15*math.sin(2*math.pi*t/6.)
    return q


def run_case(variant, count, gui, repeat, args, session):
    case_id=f'{variant}_w{count}_gui{int(gui)}_r{repeat}'
    out=FOLDER/args.result_set/case_id
    if (out/'result.json').exists()and not args.overwrite:
        print('SKIP existing',case_id,flush=True);return
    out.mkdir(parents=True,exist_ok=True)
    print(timestamp(), 'START', case_id, flush=True)
    prefix=FOLDER/'assets'/variant
    partitions=[f'so101bench_{session}_{case_id}_{i}'for i in range(count)]
    if matching_processes(partitions):raise RuntimeError('Benchmark partition collision')
    roots=[];connections=[];children=[];launches=[];gui_roots=[]
    result=dict(case_id=case_id,variant=variant,world_count=count,gui=gui,repeat=repeat,
        started_at_utc=timestamp(),status='starting',scope='common-runtime simulation assets; no ACT or PPO',
        planned_steps=args.steps,control_wall_hz=30,free_run_wall_seconds=args.free_seconds,
        world_specs=[],measurements={})
    save(out/'result.json',result)
    save(FOLDER/'progress.json',dict(case=case_id,status='starting',utc=timestamp(),result_set=args.result_set))
    monitor=Monitor(out/'resources.jsonl',roots)
    try:
        context=mp.get_context('spawn')
        for i in range(count):
            wd=out/f'world_{i:02d}';wd.mkdir(exist_ok=True)
            spec=dict(domain_id=180+i,partition=partitions[i],variant=variant,
                raw_width=1280 if variant=='baseline'else 800,raw_height=960 if variant=='baseline'else 600,
                crop_xyxy=[15,28,1215,928]if variant=='baseline'else[10,18,758,579],
                output_dir=str(wd),project_root=str(ROOT),startup_timeout_s=90)
            env=gpu_environment(prefix,spec)
            launch=spawn_log(['/opt/ros/humble/bin/ros2','launch','so101_bringup','sim.launch.py','gazebo_gui:=false','environment_randomization:=false'],env,wd/'gazebo.log')
            launches.append(launch);roots.append(launch.pid)
            if gui:
                gp=spawn_log(['/usr/bin/ign','gazebo','-g','-v','2','--gui-config',str(FOLDER/'gui.config')],env,wd/'gui.log')
                launches.append(gp);roots.append(gp.pid)
                gui_roots.append(gp.pid)
            parent,child=context.Pipe()
            worker=context.Process(target=child_entry,args=(child,spec,str(wd/'worker.log')))
            worker.start();child.close();children.append(worker);connections.append(parent);roots.append(worker.pid)
            spec['launch_pid']=launch.pid;spec['worker_pid']=worker.pid
            result['world_specs'].append(spec)
        result['startup_responses']=[receive(c,150)for c in connections]
        if gui:
            from tile_benchmark_windows import tile_windows
            result['gui_windows']=tile_windows(gui_roots)
            if len(result['gui_windows']) != count:
                raise RuntimeError('Expected one visible GUI window per world')
            if not all(w['within_work_area'] for w in result['gui_windows']):
                raise RuntimeError('Benchmark GUI windows must all be visible')
        monitor.phase='warmup'
        for step in range(args.warmup_steps):
            t=time.monotonic();command_all(connections,{'op':'step','targets':fixed_motion(step)},timeout=20)
            time.sleep(max(0,1/30-(time.monotonic()-t)))
        monitor.phase='stepped'
        begins=command_all(connections,{'op':'begin_measure'})
        begin=time.monotonic();steps=[]
        with (out/'steps.jsonl').open('w',buffering=1)as stream:
            for step in range(args.steps):
                cycle=time.monotonic()
                responses=command_all(connections,{'op':'step','targets':fixed_motion(step+args.warmup_steps)},timeout=20)
                record=dict(step=step,monotonic_s=cycle,cycle_service_s=time.monotonic()-cycle,worlds=responses)
                stream.write(json.dumps(record,default=str)+'\n');steps.append(record)
                time.sleep(max(0,1/30-(time.monotonic()-cycle)))
                if time.monotonic()-begin>args.step_wall_limit:raise TimeoutError('Stepped phase wall-time limit')
                if psutil.virtual_memory().available<3*2**30:raise MemoryError('Benchmark safety stop: available RAM below 3 GiB')
        end=time.monotonic()
        ending=command_all(connections,{'op':'end_measure'})
        result['measurements']['stepped']=dict(begin_monotonic_s=begin,end_monotonic_s=end,wall_seconds=end-begin,
            begin_responses=begins,end_responses=ending,completed_steps=len(steps))
        save(out/'result.json',result)
        if args.free_seconds>0:
            monitor.phase='free_warmup'
            command_all(connections,{'op':'resume'})
            time.sleep(3)
            monitor.phase='free_run'
            begins=command_all(connections,{'op':'begin_measure'})
            begin=time.monotonic()
            time.sleep(args.free_seconds)
            end=time.monotonic()
            ending=command_all(connections,{'op':'end_measure'})
            command_all(connections,{'op':'pause'})
            result['measurements']['free_run']=dict(begin_monotonic_s=begin,end_monotonic_s=end,wall_seconds=end-begin,
                begin_responses=begins,end_responses=ending)
        result['status']='completed'
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        # Recover partial observations without converting the failed run to a pass.
        diagnostics=[]
        for c in connections:
            try:
                if c.poll(): diagnostics.append({'pending':c.recv()})
                c.send({'op':'end_measure'});diagnostics.append(receive(c,2))
            except BaseException as secondary:diagnostics.append({'error':str(secondary)})
        result['failure_diagnostics']=diagnostics
    finally:
        monitor.phase='cleanup'
        for c in connections:
            try:c.send({'op':'close'})
            except (BrokenPipeError,EOFError,OSError):pass
        for child in children:child.join(timeout=4)
        try:result['cleaned_pids']=clean_partitions(partitions)
        except BaseException as error:result['cleanup_error']=str(error)
        for child in children:
            if child.is_alive():child.terminate();child.join(timeout=2)
            if child.is_alive():child.kill();child.join(timeout=2)
        for c in connections:c.close()
        for launch in launches:
            try:launch.wait(timeout=1)
            except subprocess.TimeoutExpired:pass
        monitor.close()
        result['ended_at_utc']=timestamp();save(out/'result.json',result)
        save(FOLDER/'progress.json',dict(case=case_id,status=result['status'],utc=timestamp(),result_set=args.result_set,error=result.get('error')))
        print(case_id,result['status'],result.get('error','')[:500],flush=True)
    if result.get('cleanup_error'):raise RuntimeError(result['cleanup_error'])
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=['baseline','latest','both'],default='both')
    p.add_argument('--worlds',type=int,nargs='+',default=[1,2,3,4])
    p.add_argument('--gui',choices=['off','on','both'],default='both')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--warmup-steps',type=int,default=60)
    p.add_argument('--free-seconds',type=float,default=20)
    p.add_argument('--step-wall-limit',type=float,default=180)
    p.add_argument('--result-set',default='trials')
    p.add_argument('--overwrite',action='store_true')
    p.add_argument('--cleanup-stopped-training',action='store_true')
    args=p.parse_args()
    if args.cleanup_stopped_training:
        active=[]
        for proc in psutil.process_iter(['pid','cmdline']):
            cmd=' '.join(proc.info['cmdline']or[])
            if 'so101_capstone.training.act_ppo_multiworld_gazebo' in cmd:active.append(proc.pid)
        if active:raise RuntimeError(f'Active training present: {active}')
        print('Removed stopped-training residue:',clean_partitions({'gz_env_23'}),flush=True)
        return
    if not os.environ.get('ROS_DISTRO'):raise RuntimeError('Source ROS and project install before starting')
    if psutil.virtual_memory().available<8*2**30:raise MemoryError('Need at least 8 GiB available before benchmark')
    session=datetime.now().strftime('%H%M%S')
    manifest=dict(started_at_utc=timestamp(),arguments=vars(args),cpu_logical_count=psutil.cpu_count(),
        cpu_physical_count=psutil.cpu_count(logical=False),ram_bytes=psutil.virtual_memory().total,
        gpu=gpu_sample(),python=sys.version,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        workload='Same current observation runtime; no target, no ACT/PPO/image augmentation; small shoulder-pan and wrist-roll sine motion around HOME. Stepped loop targets 30 wall-Hz with 33/33/34 physics steps; free run holds last command at configured RTF=1.',
        cpu_metric='Benchmark process-tree CPU, one logical core = 100%; parent sampler excluded',
        ram_metric='Sum process proportional-set-size; total RSS also retained',
        gpu_metric='Whole-device GPU/VRAM, includes desktop/background; not per-world attribution')
    manifest['cpu_model']=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),None)
    manifest['gpu_details']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
    manifest['gazebo_version']=subprocess.run(['/usr/bin/ign','gazebo','--versions'],capture_output=True,text=True).stdout.strip()
    renderer_log=Path('/home/tiger/.ignition/rendering/ogre2.log')
    manifest['renderer_preflight']=[line for line in renderer_log.read_text().splitlines() if 'GL_VENDOR =' in line or 'GL_RENDERER =' in line] if renderer_log.exists() else []
    save(FOLDER/args.result_set/'session.json',manifest)
    gui_values=[False,True]if args.gui=='both'else[args.gui=='on']
    cases=[(n,g)for n in args.worlds for g in gui_values]
    random.Random(20261002).shuffle(cases)
    variants=['baseline','latest']if args.variant=='both'else[args.variant]
    for repeat in range(1,args.repeats+1):
        for n,g in cases:
            order=variants if repeat%2 else list(reversed(variants))
            for variant in order:
                run_case(variant,n,g,repeat,args,session)
    save(FOLDER/args.result_set/'complete.json',dict(ended_at_utc=timestamp()))


if __name__=='__main__':main()
