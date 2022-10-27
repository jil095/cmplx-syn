"""
The main entry.
Usage:
python main.py -t exp_name
python main.py -a exp_name
python main.py -p exp_name
"""
import os
import argparse
from training import train_experiment, monitor_experiment
import analyzing
import plotting

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', help='CUDA device number', default=0,
                    type=int)
parser.add_argument('-t', '--train', help='Training', nargs='+', default='none')
parser.add_argument('-m', '--monitor', help='Monitoring', nargs='+', default='none')
parser.add_argument('-a', '--analyze', help='Analyzing', nargs='+', default='none')
parser.add_argument('-p', '--plot', help='Plotting', nargs='+', default='none')
parser.add_argument('-f', '--filter', help='file filtering string', type=str, default='none')
parser.add_argument('-j', '--jobs', help='num of jobs', type=str, default='1')

args = parser.parse_args()

for item in args.__dict__.items():
    print(item)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

# Training
if args.train == 'none':
    args.train = []

# Train
for name in args.train:
    train_experiment(name,)

# Monitor
if args.monitor == 'none':
    args.monitor = []

for name in args.monitor:
    monitor_experiment(name, int(args.jobs))

# Analysis
if args.analyze == 'none':
    args.analyze = []

for name in args.analyze:
    func = getattr(analyzing, name)
    if args.filter == 'none':
        func()
    else:
        func(name_filter=args.filter)

# Plot
if args.plot == 'none':
    args.plot = []

for name in args.plot:
    func = getattr(plotting, name)
    if args.filter == 'none':
        func()
    else:
        func(name_filter=args.filter)