import numpy as np
import sys
import argparse
import os
from model import *
from trainer import Trainer

parser = argparse.ArgumentParser()

# Path configurations
parser.add_argument("-savepath",
                    type=str,
                    default='out',
                    help="path to save output")
parser.add_argument("-prefix",
                    type=str,
                    default=None,
                    help="experiment prefix")
parser.add_argument("-fcnpath",
                    type=str,
                    default='/home/thanard/Downloads/FCN_mse',
                    help="path to fcn parameters for background subtraction")
parser.add_argument("-data_dir",
                    type=str,
                    default='/home/thanard/Downloads/rope_full',
                    help='path to rope_full data')
parser.add_argument("-planning_data_dir",
                    type=str,
                    default='/home/thanard/Downloads/seq_data_2',
                    help='path to seq_data_2 data')
parser.add_argument("-loadpath",
                    type=str,
                    default='',
                    help="path to the previous exp to load parameters from")
parser.add_argument("-loaditer",
                    type=int,
                    help="iteration number to load from")
parser.add_argument("-classifier_path", type=str,
                    default="classifier.pkl",
                    help="path to classifier parameters. "
                         "The classifier is pretrained on real images to classify "
                         "image pairs that are one step apart. "
                         "We use it to evaluate how feasible image transitions are,"
                         "and to select the best k plans.")

# Training hyperparameters
parser.add_argument("-seed", type=int, default=0)
parser.add_argument("-n_epochs", type=int, default=100)
parser.add_argument("-cc", type=int, default=7,
                    dest="cont_code_dim",
                    help="continuous code dimension")
parser.add_argument("-rn", type=int, default=4,
                    dest="random_noise_dim",
                    help="dimension of random noise")
parser.add_argument("-infow", type=float, default=0.1,
                    help="weight of the information term.")
parser.add_argument("-transw", type=float, default=0.1,
                    help="weight of the transition regularization term.")
parser.add_argument("-lr_d", type=float, default=0.0002)
parser.add_argument("-lr_g", type=float, default=0.0002)
# TODO: fix gtype, dtype, and k.
parser.add_argument("-gtype", type=int, default=1,
                    help="which architecture of generator to use")
parser.add_argument("-dtype", type=int, default=1,
                    help="which architecture of discriminator to use")
parser.add_argument("-qtype", type=int, default=1,
                    help="which architecture of posterior to use")
parser.add_argument("-tsize", type=int, default=[64, 64], nargs="+",
                    help="hidden size of Transition NN.")
parser.add_argument("-k", type=int, default=1,
                    help="which dataset configurations to choose from: -3 to +inf.")
parser.add_argument("-color", action="store_true")
parser.add_argument("-learn_mu", action="store_true")
parser.add_argument("-learn_var", action="store_true")

# Planning
parser.add_argument("-planning_epoch", type=int, default=[100], nargs="+",
                    help="List of epoch numbers to run planning.")
parser.add_argument("-plan_length", type=int, default=10,
                    help="Set to 0 if doesn't run planning.")
parser.add_argument("-traj_eval_copies", type=int, default=100,
                    help='the number of plans to choose from.')
parser.add_argument("-planner", type=str, default='simple_plan',
                    help="either simple_plan or astar_plan")

args = parser.parse_args()
kwargs = vars(args)

# Construct more arguments
if args.prefix is None:
    str_list = ["continuous",
                "gtype", str(args.gtype),
                "rn", str(args.rn),
                "cc", str(args.cc),
                "infow", "%.2f" % args.infow,
                "transw", "%.2f" % args.transw,
                ]
    if args.planning_horizon > 0 and os.path.exists(args.planning_data_dir) and args.planner:
        str_list.append(args.planner)
    if args.fcnpath:
        str_list.append("fcn")
    if args.learn_mu:
        str_list.append("mu")
    if args.learn_var:
        str_list.append("var")
    args.prefix = "-".join(str_list)
    print("Experiment name : ", args.prefix)
kwargs['python_cmd'] = " ".join(sys.argv)
kwargs['gray'] = not kwargs['color']
kwargs['out_dir'] = os.path.join(args.savepath, args.prefix)
if kwargs['gray'] or kwargs['fcnpath']:
    kwargs['channel_dim'] = channel_dim = 1
else:
    kwargs['channel_dim'] = channel_dim = 3

# Set initial seed
seed = kwargs['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Make output folders
out_dir = kwargs['out_dir']
for folder in ['gen', 'real-and-est', 'plans']:
    if not os.path.exists(os.path.join(out_dir, folder)):
        os.makedirs(os.path.join(out_dir, folder))

# Save configuration parameters
import json

with open('%s/params.json' % out_dir, 'w') as fp:
    json.dump(kwargs, fp, indent=4, sort_keys=True)

# Initialize Generator, Discriminator, Posterior, and FCN networks.
c_dim = kwargs['cont_code_dim']
z_dim = kwargs['random_noise_dim']

g = G(c_dim, z_dim, kwargs['gtype'], channel_dim)
d = D(kwargs['dtype'], channel_dim)
q = GaussianPosterior(c_dim, kwargs['qtype'], channel_dim)
t = GaussianTransition(c_dim,
                       hidden=kwargs['tsize'],
                       learn_var=kwargs['learn_var'],
                       learn_mu=kwargs['learn_mu'])
p = UniformDistribution(kwargs['cont_code_dim'])
var_list = [g, d, q, t, p]

if kwargs['fcnpath']:
    fcn_model = FCN_mse(n_class=2).cuda()
    fcn_model.load_state_dict(torch.load(os.path.join(kwargs['fcnpath'])))
    fcn_model.eval()
    kwargs['fcn'] = fcn_model

# Initialize or load from previously trained networks
loadpath = kwargs['loadpath']
loaditer = kwargs['loaditer']
for i in var_list:
    print(i)
    i.cuda()
    i.apply(weights_init)
    if loadpath is not None:
        if i not in [p]:
            try:
                i.load_state_dict(torch.load(os.path.join(loadpath,
                                                          'var',
                                                          '%s_%d' % (i.__class__.__name__,
                                                                     loaditer))))
                print("Loaded var %s from iter %d." % (i.__class__.__name__,
                                                       loaditer))
            except FileNotFoundError as e:
                print("Couldn't load var %s" % i.__class__.__name__)
                pass

# Training the variables
trainer = Trainer(*var_list, **kwargs)
trainer.train()
