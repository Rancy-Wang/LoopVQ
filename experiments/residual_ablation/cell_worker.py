"""Drain and resume existing work, then distribute whole benchmark/seed cells."""
import argparse
import fcntl
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from accuracy_cells import (ARMS, BASE_SHA, atomic_json, claim, initialize,
                            recipe_directory, register_done)
from run_accuracy import REPO, output, stop_owned


def command_line(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    except FileNotFoundError:
        return ''


def drain(root, arm, state_path):
    job = root / 'jobs' / arm
    marker = job / 'drain-complete.json'
    if marker.exists():
        return
    legacy = json.loads((job / 'state.json').read_text())
    parent = legacy['client_pid']
    line = command_line(parent)
    if not line:
        raise RuntimeError('Legacy client already exited; inspect its state before draining')
    assert 'run_evaluation.py' in line and str(job / 'evaluation') in line
    active = []
    for path in job.glob('evaluation/**/full/**/running.json'):
        if path.with_name('complete.json').exists():
            continue
        info = json.loads(path.read_text())
        pid = info.get('pid')
        if pid and str(path.parent) in command_line(pid):
            active.append((pid, path))
    if len(active) != 1:
        raise RuntimeError(f'Expected one owned active benchmark client: {active}')
    pid, running = active[0]
    os.kill(pid, signal.SIGSTOP)
    record = {'status': 'draining', 'arm': arm, 'paused_benchmark_pid': pid,
              'legacy_client_pid': parent, 'running_json': str(running), 'started_unix': time.time()}
    atomic_json(job / 'drain-start.json', record)
    # Only the benchmark subprocess is paused. The parent's HTTP proxy keeps
    # receiving and atomically persisting the outstanding server response.
    while True:
        if not command_line(parent):
            raise RuntimeError('HTTP proxy exited while draining; inspect saved responses')
        progress_path = job / 'server-ready.json.progress.json'
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        record.update(progress=progress, updated_unix=time.time())
        atomic_json(state_path, record)
        if progress.get('status') == 'finished':
            responses = list((running.parent / 'http/responses').glob('*.json'))
            # Verify that this specific completed request reached durable storage.
            if any(json.loads(p.read_text())['response'].get('created', 0) >=
                   int(progress['updated_unix']) for p in responses):
                record['preserved_responses_in_cell'] = len(responses)
                break
        time.sleep(5)
    assert 'run_evaluation.py' in command_line(parent)
    os.killpg(parent, signal.SIGTERM)
    # A stopped process cannot handle SIGTERM until continued. No new generation
    # is submitted because the default termination handler is already pending.
    if command_line(pid):
        os.kill(pid, signal.SIGCONT)
    deadline = time.time() + 120
    while time.time() < deadline:
        legacy = json.loads((job / 'state.json').read_text())
        if legacy.get('finished_unix'):
            break
        time.sleep(2)
    else:
        raise TimeoutError('Legacy worker did not finish its owned-process cleanup')
    record.update(status='drained', finished_unix=time.time(), legacy_final_state=legacy)
    atomic_json(marker, record)
    atomic_json(state_path, record)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--worker', required=True)
    ap.add_argument('--preferred-arm', choices=ARMS, required=True)
    ap.add_argument('--first-benchmark', default='aime24')
    ap.add_argument('--gpus', required=True)
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--drain', action='store_true')
    args = ap.parse_args()
    root = args.root.resolve()
    work = root / 'workers' / args.worker
    work.mkdir(parents=True, exist_ok=True)
    worker_lock = (work / '.lock').open('a')
    fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = work / 'state.json'
    assert os.environ.get('TMUX'), 'Use tmux for persistent evaluation'
    assert len(args.gpus.split(',')) == 2
    # All model arithmetic and benchmark/proxy code remain at the validated SHA.
    frozen = ['experiments/residual_ablation/hf_reference.py', 'calibration',
              'experiments/loopvq_reproduction/run_evaluation.py',
              'experiments/loopvq_reproduction/evaluation_io.py', 'scripts', 'tasks']
    assert not output(['git', '-C', str(REPO), 'diff', BASE_SHA, '--', *frozen])
    initialize(root)
    server = client = lease = None
    state = {'worker': args.worker, 'host': platform.node(), 'gpus': args.gpus,
             'dispatch_git_sha': output(['git', '-C', str(REPO), 'rev-parse', 'HEAD']),
             'inference_git_sha': BASE_SHA, 'started_unix': time.time()}
    loaded_arm = None
    eval_python = str(Path.home() / 'anaconda3/envs/loopvq-eval/bin/python')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, OMP_NUM_THREADS='4',
        HF_HOME=str(root.parent / '2026-11-23-reproduction/cache/huggingface'),
        HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', WANDB_BASE_URL='https://api.wandb.ai',
        WANDB_DIR=str(root / 'wandb'))
    try:
        if args.drain:
            assert args.worker == 'main-' + args.preferred_arm
            drain(root, args.preferred_arm, state_path)
        while True:
            selected = claim(root, args.worker, args.preferred_arm, args.first_benchmark)
            if selected is None:
                state['status'] = 'no_unassigned_cells'
                break
            owner, lease = selected
            arm, bench, seed = owner['arm'], owner['benchmark'], owner['seed']
            job = Path(owner['job'])
            job.mkdir(parents=True, exist_ok=True)
            state.update(status='starting_cell', cell=owner, updated_unix=time.time())
            atomic_json(state_path, state)
            if loaded_arm != arm:
                stop_owned(server)
                server = None
                busy = output(['nvidia-smi', '-i', args.gpus, '--query-compute-apps=pid', '--format=csv,noheader'])
                if busy:
                    raise RuntimeError(f'GPU pair is occupied: {args.gpus}, PIDs={busy}')
                with socket.socket() as probe:
                    probe.bind(('127.0.0.1', args.port))
                state['gpu_snapshot'] = output(['nvidia-smi', '-i', args.gpus,
                    '--query-gpu=index,uuid,name,memory.used,memory.total', '--format=csv,noheader'])
                state['host_memory'] = output(['free', '-h'])
                template = json.loads((root / 'jobs' / arm / 'server-manifest.json').read_text())
                manifest_path = job / 'server-manifest.json'
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text())
                    assert manifest['host'] == platform.node()
                    assert {line.split(',')[1].strip() for line in manifest['gpus'].splitlines()} == {
                        line.split(',')[1].strip() for line in state['gpu_snapshot'].splitlines()}
                else:
                    manifest = dict(template, host=platform.node(), gpus=state['gpu_snapshot'],
                                    replica_of=str(root / 'jobs' / arm / 'server-manifest.json'))
                    command = list(template['command'])
                    command[command.index('--port') + 1] = str(args.port)
                    command[command.index('--output') + 1] = str(job / 'server-ready.json')
                    manifest['command'] = command
                    atomic_json(manifest_path, manifest)
                command = manifest['command']
                assert command[command.index('--port') + 1] == str(args.port)
                assert manifest['inference_git_sha'] == BASE_SHA and manifest['tp'] == 2
                with (job / 'server-cell.log').open('a', buffering=1) as logfile:
                    server = subprocess.Popen(command, cwd=REPO, env=env, stdout=logfile,
                        stderr=subprocess.STDOUT, start_new_session=True)
                state['server_pid'] = server.pid
                deadline = time.time() + 600
                while time.time() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError(f'Server startup failed: {server.returncode}')
                    try:
                        with urlopen(f'http://127.0.0.1:{args.port}/health', timeout=2) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        time.sleep(2)
                else:
                    raise TimeoutError('Server startup timed out')
                loaded_arm = arm
            common = [eval_python, str(REPO / 'experiments/loopvq_reproduction/run_evaluation.py'),
                '--model', manifest['command'][manifest['command'].index('--model') + 1],
                '--base-url', f'http://127.0.0.1:{args.port}', '--method', ARMS[arm],
                '--server-manifest', str(job / 'server-manifest.json'), '--root', str(job / 'evaluation'),
                '--num-concurrent', '1', '--wandb-entity', '529513043-tsinghua-university',
                '--wandb-project', 'loopvq-ouro-2p6b-thinking', '--wandb-group', '2026-11-26-residual-ablation']
            # Existing audits are verified and replayed by the unchanged harness.
            phases = [('audit', ['--audit', '--benchmarks', 'aime24', 'evalplus_humaneval', '--seeds', '0', '1']),
                      ('full', ['--benchmarks', bench, '--seeds', str(seed)])]
            for phase, tail in phases:
                state.update(status=phase, command=common + tail, updated_unix=time.time())
                with (work / f'{arm}-{bench}-{seed}-{phase}.log').open('a', buffering=1) as logfile:
                    client = subprocess.Popen(common + tail, cwd=REPO, env=env, stdout=logfile,
                        stderr=subprocess.STDOUT, start_new_session=True)
                state['client_pid'] = client.pid
                atomic_json(state_path, state)
                while client.poll() is None:
                    if server.poll() is not None:
                        raise RuntimeError(f'Server exited during {phase}: {server.returncode}')
                    time.sleep(5)
                if client.returncode:
                    raise RuntimeError(f'{phase} client failed: {client.returncode}')
            result = register_done(root, owner)
            print('CELL_COMPLETE', arm, bench, seed, result['scores'], flush=True)
            lease.close()
            lease = None
        state['exit_code'] = 0
    except BaseException as exc:
        state.update(status='failed', exit_code=1, error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        stop_owned(client)
        stop_owned(server)
        if lease:
            lease.close()
        state['finished_unix'] = time.time()
        atomic_json(state_path, state)


if __name__ == '__main__':
    main()
