"""Derive job-local Ray ports; check ownership without killing any process."""

import argparse
import json
import os
import shlex
import socket


def ports(job):
    job = int(job)
    if job <= 0:
        raise ValueError('positive SLURM_JOB_ID required')
    offset = job % 10000
    return {
        'RECIPE_RAY_GCS_PORT': 20000 + offset,
        'RECIPE_RAY_CLIENT_PORT': 30000 + offset,
    }


def check_free(selected):
    held = []
    try:
        for port in selected.values():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            held.append(sock)
            # No SO_REUSEADDR: a residual listener must fail before warmup.
            sock.bind(('0.0.0.0', port))
    finally:
        for sock in held:
            sock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    selected = ports(os.environ['SLURM_JOB_ID'])
    if args.check:
        check_free(selected)
        print(json.dumps({'ray_ports_free': selected}), flush=True)
    else:
        selected['RAY_ADDRESS'] = (
            f"{os.environ['RAY_HEAD_IP']}:{selected['RECIPE_RAY_GCS_PORT']}"
        )
        for key, value in selected.items():
            print(f'export {key}={shlex.quote(str(value))}')


if __name__ == '__main__':
    main()
