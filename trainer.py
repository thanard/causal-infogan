import numpy as np
import doodad as dd
import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tensorboard_logger import configure, log_value
from collections import OrderedDict

from planning import plan_traj_astar, discretize, undiscretize
from dataset import ImagePairs
from utils import plot_img, from_numpy_to_var, print_array, write_number_on_images, write_stats_from_var
from model import get_causal_classifier
from logger import Logger


class Trainer:
    def __init__(self, G, FE, D, Q, T, P, V=None, **kwargs):
        self.P = P
        self.G = G
        self.FE = FE
        self.D = D
        self.Q = Q
        self.T = T
        self.V = V
        self.k = kwargs['k']
        self.q_eval = kwargs['q_eval']
        self.gray = kwargs['gray']
        self.red = kwargs['red']
        self.n_epochs = kwargs['n_epochs']
        self.dis_c_dim = kwargs['disc_code_dim']
        self.con_c_dim = kwargs['cont_code_dim']
        self.rand_z_dim = kwargs['random_noise_dim']
        self.planning_iter = kwargs['planning_iter']
        self.planning_horizon = kwargs['planning_horizon']
        self.discretization_bins = kwargs['discretization_bins']
        self.traj_eval_copies = kwargs['traj_eval_copies']
        self.infow = kwargs['infow']
        self.transw = kwargs['transw']
        self.vaew = kwargs['vaew']
        self.auxw = kwargs['auxw']
        self.lr_g = kwargs['lr_g']
        self.lr_d = kwargs['lr_d']
        self.fcn = kwargs.get('fcn', None)
        self.channel_dim = kwargs['channel_dim']
        self.c_dim = self.dis_c_dim + self.con_c_dim
        self.latent_dim = self.con_c_dim + 2*self.dis_c_dim + self.rand_z_dim
        self.load_D = get_causal_classifier(kwargs['load_D'], default=D)

        self.batch_size = 100
        self.test_sample_size = 12
        # self.test_num_codes = 10
        self.test_num_codes = max(20, self.con_c_dim + 1)
        self.test_size = self.test_sample_size * self.test_num_codes

        self.out_dir = dd.get_args('output_dir')
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        self.eval_input = self._eval_noise()

        # No longer use logger.
        self.configure_logger()
        self.log_dict = OrderedDict()

        self.planner = getattr(self, kwargs['planner'])

    def configure_logger(self):
        self.logger = Logger(os.path.join(self.out_dir, "var_log"))
        configure(os.path.join(self.out_dir, "var_log"), flush_secs=5)

    def _noise_sample(self, z, bs):
        c = self.P.sample(bs)
        c_next = self.T(c)
        z.data.normal_(0, 1)
        return z, c, c_next

    def _eval_noise(self):
        '''
        :return: z (sample_size x num_codes x z_dim), c (sample_size x num_codes x z_dim)
        '''
        assert self.dis_c_dim == 0
        more_codes = self.test_num_codes - (self.c_dim + 1)
        # c = Variable(torch.cuda.FloatTensor([[j<i for j in range(self.disc_c_dim)] for i in range(min(self.test_num_codes, self.disc_c_dim+1))]))
        c = Variable(torch.cuda.FloatTensor([[j<i for j in range(self.con_c_dim)] for i in range(min(self.test_num_codes, self.con_c_dim+1))]))*(self.P.unif_range[1] - self.P.unif_range[0]) + self.P.unif_range[0]
        if more_codes > 0:
            c = torch.cat([c, self.P.sample(more_codes)], 0)
        self.eval_c = c
        z = Variable(torch.FloatTensor(self.test_sample_size, self.rand_z_dim).normal_(0, 1).cuda())

        plot_img(c.t().data,
                 os.path.join(self.out_dir, 'eval_code.png'),
                 vrange=self.P.unif_range)
        return z[:, None, :].repeat(1, self.test_num_codes, 1).view(-1, self.rand_z_dim), \
               c.repeat(1, 1, self.test_sample_size).permute(2, 0, 1).contiguous().view(-1, self.con_c_dim), \

    def get_c_next(self, epoch):
        c_next = self.T(self.eval_c)
        plot_img(c_next.t().data,
                 os.path.join(self.out_dir, 'eval_code_next_%d.png' % epoch),
                 vrange=self.P.unif_range)
        return c_next.repeat(1, 1, self.test_sample_size).permute(2, 0, 1).contiguous().view(-1, self.con_c_dim)

    # def save_plots(self, trans_prob, real_prob, path):
    #     import matplotlib
    #     matplotlib.use('Agg')
    #     import matplotlib.pyplot as plt
    #     import matplotlib.gridspec as gridspec
    #     from causal_infogan_rope_continuous.utils import hsv_transition
    #     c_dim = trans_prob.shape[0]
    #     cmap = matplotlib.cm.get_cmap('hsv')
    #     norm = matplotlib.colors.Normalize(vmin=0.0, vmax=c_dim)
    #
    #     fig = plt.figure(figsize=(6, 3))
    #     gs = gridspec.GridSpec(1, 2)
    #
    #     plt.subplot(gs[0, 0])
    #     plt.title('Transition raw')
    #     plt.imshow(hsv_transition(trans_prob, c_dim, cmap, norm), aspect='auto')
    #
    #     plt.subplot(gs[0, 1])
    #     plt.title('Real transition prob')
    #     plt.imshow(hsv_transition(real_prob, c_dim, cmap, norm), aspect='auto')
    #     # Apply Q to cluster real images.
    #     # plt.subplot(gs[0, 1])
    #     # plt.title('Q eval on real images')
    #     # plt.imshow()
    #     # plt.imshow()
    #     plt.savefig(path)
    #     plt.close()

    def apply_fcn(self, img):
        assert self.fcn
        return torch.sign(F.softmax(self.fcn(Variable(img).cuda()))[:, 0, None, :, :].detach() - 0.5)

    def apply_fcn_mse(self, img):
        o = self.fcn(Variable(img).cuda()).detach()
        return torch.clamp(2*(o - 0.5), -1 + 1e-3, 1 - 1e-3)
        # return torch.clamp(2.6*(o - 0.5), -1 + 1e-3, 1 - 1e-3)

    def preprocess_function(self, state):
        # import ipdb; ipdb.set_trace()
        # q_eval = self.Q.forward_hard(self.FE(obs)).detach().cpu().numpy()
        return discretize(state, self.discretization_bins, self.P.unif_range)

    def discriminator_function(self, obs, obs_next):
        # out = d.forward(from_numpy_to_var(obs), from_numpy_to_var(obs_next))
        out = self.load_D(self.FE(obs), self.FE(obs_next))
        # out = self.D.forward(self.FE(obs), self.FE(obs_next))
        return out.detach().cpu().numpy()

    def discriminator_function_np(self, obs, obs_next):
        return self.discriminator_function(from_numpy_to_var(obs),
                                           from_numpy_to_var(obs_next))

    def continuous_transition_function(self, c_):
        c_ = undiscretize(c_, self.discretization_bins, self.P.unif_range)
        c_next_ = self.T(from_numpy_to_var(c_)).data.cpu().numpy()
        c_next_ = np.clip(c_next_, self.P.unif_range[0] + 1e-6, self.P.unif_range[1] - 1e-6)
        c_next_d = discretize(c_next_, self.discretization_bins, self.P.unif_range)
        return c_next_d

    def conditional_generator_function(self, c_, c_next_, obs):
        '''
        This doesn't do anything.
        '''
        c_ = undiscretize(c_, self.discretization_bins, self.P.unif_range)
        c_next_ = undiscretize(c_next_, self.discretization_bins, self.P.unif_range)
        z_ = from_numpy_to_var(np.random.randn(c_.shape[0], self.rand_z_dim))
        _, next_observation = self.G(z_, from_numpy_to_var(c_), from_numpy_to_var(c_next_))
        return next_observation.data.cpu().numpy()

    def train(self):
        # Set up training.
        real_o = Variable(torch.FloatTensor(self.batch_size, 3, 64, 64).cuda(), requires_grad=False)
        real_o_next = Variable(torch.FloatTensor(self.batch_size, 3, 64, 64).cuda(), requires_grad=False)
        label = Variable(torch.FloatTensor(self.batch_size).cuda(), requires_grad=False)
        z = Variable(torch.FloatTensor(self.batch_size, self.rand_z_dim).cuda(), requires_grad=False)

        criterionD = nn.BCELoss().cuda()
        # criterionD = nn.MSELoss().cuda()
        l1loss = nn.L1Loss()

        optimD = optim.Adam([{'params': self.FE.parameters()}, {'params': self.D.parameters()}], lr=self.lr_d,
                            betas=(0.5, 0.999))
        optimG = optim.Adam([{'params': self.G.parameters()},
                             {'params': self.Q.parameters()},
                             {'params': self.T.parameters()}], lr=self.lr_g,
                            betas=(0.5, 0.999))

        # Load dataset.
        rope_path = dd.get_args('data_dir')
        rope_path = os.path.realpath(rope_path)

        dataloader = None
        trans = [
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

        if not self.fcn:
            # If fcn it will do the transformation to gray
            # and normalize in the loop.
            trans.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            if self.gray:
                # Apply grayscale transformation.
                if self.red:
                    trans.append(lambda x: (x[0])[None, :, :])
                else:
                    trans.append(lambda x: x.mean(dim=0)[None, :, :])

        # if self.k >= -2:
        #     # Rope dataset only.
        #     mask = torch.ones(64, 64)
        #     mask[0:6, :] = 0.0
        #     for x in range(5, 36):
        #         for y in range(48, 64):
        #             if 2 * y >= x + 91:
        #                 mask[x, y] = 0.0
        #     trans.append(lambda x: x * mask.repeat(self.channel_dim, 1, 1))


        trans_comp = transforms.Compose(trans)
        if self.k == -3:
            # MNIST
            # Randomly sample the first and the second images.
            # TODO: remove local path hack.
            assert self.gray
            imgs_1 = dset.MNIST(root="/home/thanard/Downloads/mnist",
                                transform=trans_comp, download=dd.get_args('ec2'))
            imgs_2 = dset.MNIST(root="/home/thanard/Downloads/mnist",
                                transform=trans_comp)
            dataloader1 = torch.utils.data.DataLoader(imgs_1,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      num_workers=1,
                                                      drop_last=True)
            dataloader2 = torch.utils.data.DataLoader(imgs_2,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      num_workers=1,
                                                      drop_last=True)
        elif self.k == -2:
            # Rope dataset
            # One rope and one random images.
            imgs_1 = dset.ImageFolder(root=rope_path,
                                      transform=trans_comp)
            imgs_2 = dset.ImageFolder(root=rope_path,
                                      transform=lambda x: torch.randn(3, 64, 64))
            dataloader1 = torch.utils.data.DataLoader(imgs_1,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      num_workers=1,
                                                      drop_last=True)
            dataloader2 = torch.utils.data.DataLoader(imgs_2,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      num_workers=1,
                                                      drop_last=True)
        elif self.k == -1:
            # Rope dataset
            # Two random rope images.
            imgs_1 = dset.ImageFolder(root=rope_path,
                                      transform=trans_comp)
            imgs_2 = dset.ImageFolder(root=rope_path,
                                      transform=trans_comp)
            dataloader1 = torch.utils.data.DataLoader(imgs_1,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      num_workers=1,
                                                      drop_last=True)
            dataloader2 = torch.utils.data.DataLoader(imgs_2,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      num_workers=1,
                                                      drop_last=True)
        elif self.k >= 0:
            # Rope dataset
            # Image 1 and image 2 are k steps apart.
            dataset = ImagePairs(root=rope_path,
                                 transform=trans_comp,
                                 n_frames_apart=self.k)
            dataloader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=True,
                                                     num_workers=2,
                                                     drop_last=True)
        else:
            raise NotImplementedError
        ############################################
        # Load eval plan dataset
        planning_data_dir = dd.get_args('planning_data_dir')
        dataset_start = dset.ImageFolder(root=os.path.join(planning_data_dir, 'start'),
                                         transform=trans_comp)
        dataset_goal = dset.ImageFolder(root=os.path.join(planning_data_dir, 'goal'),
                                         transform=trans_comp)
        data_start_loader = torch.utils.data.DataLoader(dataset_start,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        num_workers=1,
                                                        drop_last=True)
        data_goal_loader = torch.utils.data.DataLoader(dataset_goal,
                                                       batch_size=1,
                                                       shuffle=False,
                                                       num_workers=1,
                                                       drop_last=True)
        ############################################
        for epoch in range(self.n_epochs + 1):
            # trans_prob = np.zeros((2 ** self.dis_c_dim, 2 ** self.dis_c_dim))
            # real_trans = np.zeros((2 ** self.dis_c_dim, 2 ** self.dis_c_dim))
            # Let's assume that batch_data has image pairs of transitions.
            self.Q.train()
            self.FE.train()
            self.D.train()
            self.T.train()
            self.G.train()
            if self.k < 0:
                dataloader = zip(dataloader1, dataloader2)
            for num_iters, batch_data in enumerate(dataloader, 0):
                # D training
                # Real data
                optimD.zero_grad()
                o, _ = batch_data[0]
                o_next, _ = batch_data[1]
                bs = o.size(0)

                real_o.data.resize_(o.size())
                real_o_next.data.resize_(o_next.size())
                label.data.resize_(bs)

                real_o.data.copy_(o)
                real_o_next.data.copy_(o_next)
                if self.fcn:
                    real_o = self.apply_fcn_mse(o)
                    real_o_next = self.apply_fcn_mse(o_next)
                    if real_o.abs().max() > 1:
                        import ipdb; ipdb.set_trace()
                    assert real_o.abs().max() <= 1

                if epoch == 0:
                    break

                real_x = self.FE(real_o)
                real_x_next = self.FE(real_o_next)
                probs_real = self.D(real_x, real_x_next)
                label.data.fill_(1)
                loss_real = criterionD(probs_real, label)
                loss_real.backward()

                # Fake data
                z, c, c_next = self._noise_sample(z, bs)
                fake_o, fake_o_next = self.G(z, c, c_next)
                fake_x = self.FE(fake_o.detach())
                fake_x_next = self.FE(fake_o_next.detach())
                probs_fake = self.D(fake_x, fake_x_next, True)
                label.data.fill_(0)
                loss_fake = criterionD(probs_fake, label)
                loss_fake.backward()

                D_loss = loss_real + loss_fake

                optimD.step()
                ############################################
                # G loss (Update G)
                optimG.zero_grad()

                fake_x = self.FE(fake_o)
                fake_x_next = self.FE(fake_o_next)
                probs_fake_2 = self.D(fake_x, fake_x_next)
                label.data.fill_(1)
                G_loss = criterionD(probs_fake_2, label)

                # Q loss (Update G, T, Q)
                ent_loss = -self.P.log_prob(c).mean(0)
                crossent_loss = -self.Q.log_prob(fake_x, c).mean(0)
                crossent_loss_next = -self.Q.log_prob(fake_x_next, c_next).mean(0)
                # trans_prob = self.T.get_prob(Variable(torch.eye(self.dis_c_dim).cuda()))
                ent_loss_next = -self.T.log_prob(c, None, c_next).mean(0)
                mi_loss = crossent_loss - ent_loss
                mi_loss_next = crossent_loss_next - ent_loss_next
                Q_loss = mi_loss + mi_loss_next

                # T loss (Update T)
                # TODO: fix the variance computation when mean is not 0.
                Q_c_given_x, Q_c_given_x_var = (i.detach() for i in self.Q.forward(real_x))
                Q_c_given_x_next = self.Q.forward_hard(real_x_next)
                t_mu, t_variance = self.T.get_mu_and_var(c)
                t_diff = t_mu - c
                # T_loss = ((t_variance - (Q_c_given_x_next - Q_c_given_x)**2)**2).sum(1).mean(0)
                # Let's just keep the variance small.
                T_loss = (t_variance**2).sum(1).mean(0)

                # self.Q.train()
                # T_loss = (Q_c_given_x_next - self.T(Q_c_given_x)).pow(2)
                # T_loss = torch.mean(
                #     torch.sum(torch.mean(
                #         binary_cross_entropy(self.T.forward_all_actions(Q_c_given_x),
                #                              Q_c_given_x_next[:, None, :]), dim=1),
                #         dim=1)
                # )
                # T_loss = Variable(torch.FloatTensor([0.]).cuda())

                # VAE loss (Update V)
                # TODO: train on best z.
                z.data.normal_(0, 1)
                est_o, est_o_next = self.G(z, Q_c_given_x, Q_c_given_x_next)
                diff_o = l1loss(est_o, real_o)
                diff_o_next = l1loss(est_o_next, real_o_next)
                VAE_loss = diff_o + diff_o_next

                # G loss 2 (Update G)
                probs_fake_3 = self.D(real_x, self.FE(est_o_next))
                # probs_fake_3_next = self.D(self.FE(est_o), real_x_next)
                Aux_loss = criterionD(probs_fake_3, label)

                (G_loss +
                 self.infow * Q_loss +
                 self.transw * T_loss +
                 self.vaew * VAE_loss +
                 self.auxw * Aux_loss).backward()
                optimG.step()
                #############################################
                # # Update real transitions
                # c_ind = binary_to_int(Q_c_given_x.data.cpu().numpy(),
                #                       width=self.dis_c_dim).astype(int)
                # c_next_ind = binary_to_int(Q_c_given_x_next.data.cpu().numpy(),
                #                            width=self.dis_c_dim).astype(int)
                # real_trans[c_ind, c_next_ind] += 1
                # # Update estimated transition
                # c_ind = binary_to_int(c.data.cpu().numpy(),
                #                       width=self.dis_c_dim).astype(int)
                # c_next_ind = binary_to_int(c_next.data.cpu().numpy(),
                #                            width=self.dis_c_dim).astype(int)
                # trans_prob[c_ind, c_next_ind] += 1
                #############################################
                if num_iters % 100 == 0:
                    self.log_dict['Dloss'] = D_loss.data[0]
                    self.log_dict['Gloss'] = G_loss.data[0]
                    self.log_dict['Qloss'] = Q_loss.data[0]
                    self.log_dict['Tloss'] = T_loss.data[0]
                    self.log_dict['VAE_loss'] = VAE_loss.data[0]
                    self.log_dict['Aux_loss'] = Aux_loss.data[0]
                    self.log_dict['diff_o'] = diff_o.data[0]
                    self.log_dict['diff_o_next'] = diff_o_next.data[0]
                    self.log_dict['mi_loss'] = mi_loss.data[0]
                    self.log_dict['mi_loss_next'] = mi_loss_next.data[0]
                    self.log_dict['ent_loss'] = ent_loss.data[0]
                    self.log_dict['ent_loss_next'] = ent_loss_next.data[0]
                    self.log_dict['crossent_loss'] = crossent_loss.data[0]
                    self.log_dict['crossent_loss_next'] = crossent_loss_next.data[0]
                    self.log_dict['D(real)'] = probs_real.data.mean()
                    self.log_dict['D(fake)_before'] = probs_fake.data.mean()
                    self.log_dict['D(fake)_after'] = probs_fake_2.data.mean()
                    self.log_dict['D(real, fake)_after'] = probs_fake_3.data.mean()

                    write_stats_from_var(self.log_dict, Q_c_given_x, 'Q_c_given_real_x_mu')
                    write_stats_from_var(self.log_dict, Q_c_given_x, 'Q_c_given_real_x_mu', idx=0)
                    write_stats_from_var(self.log_dict, Q_c_given_x_var, 'Q_c_given_real_x_variance')
                    write_stats_from_var(self.log_dict, Q_c_given_x_var, 'Q_c_given_real_x_variance', idx=0)

                    write_stats_from_var(self.log_dict, t_mu, 't_mu')
                    write_stats_from_var(self.log_dict, t_mu, 't_mu', idx=0)
                    write_stats_from_var(self.log_dict, t_diff, 't_diff')
                    write_stats_from_var(self.log_dict, t_diff, 't_diff', idx=0)
                    write_stats_from_var(self.log_dict, t_variance, 't_variance')
                    write_stats_from_var(self.log_dict, t_variance, 't_variance', idx=0)

                    print('\n#######################'
                          '\nEpoch/Iter:%d/%d; '
                          '\nDloss: %.3f; '
                          '\nGloss: %.3f; '
                          '\nQloss: %.3f, %.3f; '
                          '\nT_loss: %.3f; '
                          '\nAux_G_loss: %.3f; '
                          '\nL1loss: %.3f; '
                          '\nEnt: %.3f, %.3f; '
                          '\nCross Ent: %.3f, %.3f; '
                          '\nD(x): %.3f; '
                          '\nD(G(z)): b %.3f, a %.3f;'
                          '\nD(x, GQx): %.3f'
                          '\n0_Q_c_given_rand_x_mean: %.3f'
                          '\n0_Q_c_given_rand_x_std: %.3f'
                          '\n0_Q_c_given_fixed_x_std: %.3f'
                          '\nt_diff_abs_mean: %.3f'
                          '\nt_std_mean: %.3f'
                          % (epoch, num_iters,
                             D_loss.data[0],
                             G_loss.data[0],
                             mi_loss.data[0], mi_loss_next.data[0],
                             T_loss.data[0],
                             Aux_loss.data[0],
                             VAE_loss.data[0],
                             ent_loss.data[0], ent_loss_next.data[0],
                             crossent_loss.data[0], crossent_loss_next.data[0],
                             probs_real.data.mean(),
                             probs_fake.data.mean(), probs_fake_2.data.mean(),
                             probs_fake_3.data.mean(),
                             Q_c_given_x[:, 0].data.mean(),
                             Q_c_given_x[:, 0].data.std(),
                             np.sqrt(Q_c_given_x_var[:, 0].data.mean()),
                             t_diff.data.abs().mean(),
                             t_variance.data.sqrt().mean(),
                             ))
            # if epoch % 1 == 0:
            #     import ipdb; ipdb.set_trace()
            #############################################
            # Start evaluation from here.
            self.Q.eval()
            self.FE.eval()
            self.D.eval()
            self.T.eval()
            self.G.eval()
            # Plot fake data
            x_save, x_next_save = self.G(*self.eval_input, self.get_c_next(epoch))
            # save_image(torch.cat([x_save, x_next_save, x_save-x_next_save], dim=0).data,
            #            os.path.join(self.out_dir, 'gen_samples_%d_%03d.png' % (i + 1, epoch)),
            #            nrow=self.test_sample_size,
            #            normalize=True,
            #            scale_each=True)
            save_image(x_save.data,
                       os.path.join(self.out_dir, 'gen', 'curr_samples_%03d.png' % epoch),
                       nrow=self.test_num_codes,
                       normalize=True)
            save_image(x_next_save.data,
                       os.path.join(self.out_dir, 'gen', 'next_samples_%03d.png' % epoch),
                       nrow=self.test_num_codes,
                       normalize=True)
            save_image((x_save-x_next_save).data,
                       os.path.join(self.out_dir, 'gen', 'diff_samples_%03d.png' % epoch),
                       nrow=self.test_num_codes,
                       normalize=True)
            #############################################
            # Plot real data.
            if epoch % 10 == 0:
                save_image(real_o.data,
                           os.path.join(self.out_dir, 'real-and-est', 'real_samples_%d.png' % epoch),
                           nrow=self.test_num_codes,
                           normalize=True)
                # save_image((real_o-real_o_next).data,
                #            os.path.join(self.out_dir, 'real-and-est', 'real_samples_diff_%d.png' % _label),
                #            nrow=self.test_num_codes,
                #            normalize=True)
                save_image(real_o_next.data,
                           os.path.join(self.out_dir, 'real-and-est', 'real_samples_next_%d.png' % epoch),
                           nrow=self.test_num_codes,
                           normalize=True)
            #############################################
            # Show estimated data.
            try:
                save_image(est_o.data,
                           os.path.join(self.out_dir, 'real-and-est', 'est_samples_%d.png' % _label),
                           nrow=self.test_num_codes,
                           normalize=True)
                save_image(est_o_next.data,
                           os.path.join(self.out_dir, 'real-and-est', 'est_samples_next_%d.png' % _label),
                           nrow=self.test_num_codes,
                           normalize=True)
            except:
                pass
            #############################################
            # Plot transition
            # self.save_plots(normalize_row(trans_prob)*2**(self.dis_c_dim-4),
            #                 normalize_row(real_trans)*2**(self.dis_c_dim-4),
            #                 os.path.join(self.out_dir, 'trans_prob_%03d.png' % epoch))
            #############################################
            # Save parameters
            if epoch % 5 == 0:
                if not os.path.exists('%s/var' % self.out_dir):
                    os.makedirs('%s/var' % self.out_dir)
                for i in [self.FE, self.D, self.G, self.Q, self.T]:
                    torch.save(i.state_dict(),
                               os.path.join(self.out_dir,
                                            'var',
                                            '%s_%d' % (i.__class__.__name__, epoch,
                                            )))
                # import pickle
                # with open(os.path.join(self.out_dir, 'var/real_transition_%d.pkl' % epoch), 'wb') as f:
                #     pickle.dump(real_trans, f)
                # with open(os.path.join(self.out_dir, 'var/est_transition_%d.pkl' % epoch), 'wb') as f:
                #     pickle.dump(trans_prob, f)
            #############################################
            for k, v in self.log_dict.items():
                log_value(k, v, epoch)

            if epoch > 0:
                # tf logger
                # log_value('avg|x_next - x|', (x_next_save.data - x_save.data).abs().mean(dim=0).sum(), epoch + 1)
                # self.logger.histo_summary("Q_c_given_x", Q_c_given_x.data.cpu().numpy().reshape(-1), step=epoch)
                # self.logger.histo_summary("Q_c0_given_x", Q_c_given_x[:, 0].data.cpu().numpy(), step=epoch)
                # self.logger.histo_summary("Q_c_given_x_var", Q_c_given_x_var.cpu().numpy().reshape(-1), step=epoch)
                # self.logger.histo_summary("Q_c0_given_x_var", Q_c_given_x_var[:, 0].data.cpu().numpy(), step=epoch)

                # csv log
                with open(os.path.join(self.out_dir, 'progress.csv'), 'a') as csv_file:
                    writer = csv.writer(csv_file)
                    # import ipdb; ipdb.set_trace()
                    if epoch == 1:
                        writer.writerow(["epoch"] + list(self.log_dict.keys()))
                    writer.writerow(["%.3f" % _tmp for _tmp in [epoch] + list(self.log_dict.values())])
            #############################################
            # Do planning?
            if self.planning_horizon <= 0 or epoch not in self.planning_iter:
                continue
            if self.G.gtype not in [70, 71, 6, 92]:
                continue
            print("\n#######################"
                  "\nPlanning")
            #############################################
            # # Showing plans on real images using Q.
            # planning_dataloader = zip(data_start_loader, data_goal_loader)
            # for i, pair in enumerate(planning_dataloader, 0):
            #     if self.fcn:
            #         start_obs = self.apply_fcn_mse(pair[0][0])
            #         goal_obs = self.apply_fcn_mse(pair[1][0])
            #
            #     c_start = self.Q.forward_hard(self.FE(start_obs)).repeat(self.traj_eval_copies, 1)
            #     c_goal = self.Q.forward_hard(self.FE(goal_obs)).repeat(self.traj_eval_copies, 1)
            #     rollout = self.plan(c_start, c_goal)
            #     # rollout = []
            #     rollout.insert(0, start_obs.repeat(self.traj_eval_copies, 1, 1, 1))
            #     rollout.append(goal_obs.repeat(self.traj_eval_copies, 1, 1, 1))
            #
            #     rollout_best_k, confidences = self.get_best_k(rollout, keep_best)
            #     rollout_data = torch.stack(rollout_best_k, dim=0)
            #
            #     masks = - np.ones((rollout_data.size()[0], keep_best, self.channel_dim, 64, 64),
            #                     dtype=np.float32)
            #     write_number_on_images(masks, confidences)
            #
            #     save_image(torch.max(rollout_data, from_numpy_to_var(masks)).view(-1, self.channel_dim, 64, 64).data,
            #                os.path.join(self.out_dir, 'plans', 'rollout_Q_%d_epoch_%d.png' % (i, epoch)),
            #                nrow=keep_best,
            #                normalize=True)
            #############################################
            # Showing plans on real images using best code.
            # Min l2 distance from start and goal real images.
            self.plan_hack(data_start_loader,
                      data_goal_loader,
                      epoch,
                      'L2')

            # Min classifier distance from start and goal real images.
            self.plan_hack(data_start_loader,
                      data_goal_loader,
                      epoch,
                      'load_D')
            #############################################
            # # Showing plans on generated images using random code.
            # c_start = self.eval_c[:self.con_c_dim]
            # c_goal = self.eval_c[1:self.con_c_dim+1]
            # rollout = self.plan(c_start, c_goal)
            # save_image(torch.cat(rollout, dim=0).data,
            #            os.path.join(self.out_dir, 'plans', 'fake_epoch_%d.png' % epoch),
            #            nrow=self.con_c_dim,
            #            normalize=True)
            #############################################
            # # Showing plans on generated images using Q
            # o_start = rollout[0]
            # o_goal = rollout[-1]
            # c_start = self.Q.forward_hard(self.FE(o_start))
            # c_goal = self.Q.forward_hard(self.FE(o_goal))
            # rollout = self.plan(c_start, c_goal)
            # rollout.insert(0, o_start)
            # rollout.append(o_goal)
            # save_image(torch.cat(rollout, dim=0).data,
            #            os.path.join(self.out_dir, 'plans', 'fake_Q_epoch_%d.png' % epoch),
            #            nrow=self.con_c_dim,
            #            normalize=True)
            # # if epoch % 5 == 0:
            # #     import ipdb; ipdb.set_trace()

    def plan_hack(self,
                  data_start_loader,
                  data_goal_loader,
                  epoch,
                  metric,
                  keep_best=10):
        all_confidences = []
        c_start = None
        est_start_obs = None
        for img in data_start_loader:
            if self.fcn:
                start_obs = self.apply_fcn_mse(img[0])
            else:
                start_obs = img[0]
            pt_start = os.path.join(self.out_dir, 'plans', 'c_min_start_%s.pt' % metric)
            if os.path.exists(pt_start):
                z_start, c_start, _, est_start_obs = torch.load(pt_start)
            else:
                z_start, c_start, _, est_start_obs = self.closest_code(start_obs,
                                                                 400,
                                                                 False,
                                                                 metric, 1)
                torch.save([z_start, c_start, _, est_start_obs], pt_start)
            break
        # Hacky for now
        try:
            c_start = Variable(c_start)
            est_start_obs = Variable(est_start_obs)
        except RuntimeError:
            pass

        for i, img in enumerate(data_goal_loader, 0):
            if self.fcn:
                goal_obs = self.apply_fcn_mse(img[0])
            else:
                goal_obs = img[0]
            pt_goal = os.path.join(self.out_dir, 'plans', 'c_min_goal_%s_%d_epoch_%d.pt' % (metric, i, epoch))
            if os.path.exists(pt_goal):
                z_goal, _, c_goal, est_goal_obs = torch.load(pt_goal)
            else:

                z_goal, _, c_goal, est_goal_obs = self.closest_code(goal_obs,
                                                               400,
                                                               True,
                                                               metric, 1)
                torch.save([z_goal, _, c_goal, est_goal_obs], pt_goal)
            # Hacky for now
            try:
                c_goal = Variable(c_goal)
                est_goal_obs = Variable(est_goal_obs)
            except RuntimeError:
                pass
            # Plan using c_start and c_goal.
            rollout = self.planner(c_start.repeat(self.traj_eval_copies, 1),
                                   c_goal.repeat(self.traj_eval_copies, 1),
                                   start_obs=start_obs,
                                   goal_obs=goal_obs)

            # Insert closest start and goal.
            rollout.insert(0, est_start_obs.repeat(self.traj_eval_copies, 1, 1, 1))
            rollout.append(est_goal_obs.repeat(self.traj_eval_copies, 1, 1, 1))

            # Insert real start and goal.
            rollout.insert(0, start_obs.repeat(self.traj_eval_copies, 1, 1, 1))
            rollout.append(goal_obs.repeat(self.traj_eval_copies, 1, 1, 1))

            rollout_best_k, confidences = self.get_best_k(rollout, keep_best)
            rollout_data = torch.stack(rollout_best_k, dim=0)

            masks = - np.ones((rollout_data.size()[0], keep_best, self.channel_dim, 64, 64),
                              dtype=np.float32)
            write_number_on_images(masks, confidences)

            # save_image(torch.max(rollout_data, from_numpy_to_var(masks)).view(-1, self.channel_dim, 64, 64).data,
            #            os.path.join(self.out_dir, 'plans', '%s_min_%s_%d_epoch_%d.png'
            #                         % (self.planner.__name__, metric, i, epoch)),
            #            nrow=keep_best,
            #            normalize=True)

            pd = torch.max(rollout_data, from_numpy_to_var(masks)).t().contiguous().view(-1, self.channel_dim,
                                                                                                  64, 64)
            # confidences.T has size keep_best x rollout length
            all_confidences.append(confidences.T[-1][:-1])

            save_image(pd.data,
                       os.path.join(self.out_dir, 'plans', '%s_min_%s_%d_epoch_%d.png'
                                    % (self.planner.__name__, metric, i, epoch)),
                       nrow=int(pd.size()[0] / keep_best),
                       normalize=True)
        all_confidences = np.stack(all_confidences)
        print((all_confidences[:, 0]>0.9).sum(), (all_confidences[:, -1]>0.9).sum())
        import pickle as pkl
        with open(os.path.join(self.out_dir, 'all_confidences.pkl'), 'wb') as f:
            pkl.dump(all_confidences, f)
        import matplotlib.pyplot as plt
        plt.boxplot([all_confidences.mean(1), all_confidences[all_confidences[:, -1]>0.9].mean(1)])
        plt.savefig(os.path.join(self.out_dir, 'boxplot.png'))

    def plan(self,
             data_start_loader,
             data_goal_loader,
             epoch,
             metric,
             keep_best=10):
        planning_dataloader = zip(data_start_loader, data_goal_loader)
        for i, pair in enumerate(planning_dataloader, 0):
            if self.fcn:
                start_obs = self.apply_fcn_mse(pair[0][0])
                goal_obs = self.apply_fcn_mse(pair[1][0])

            # Compute c_start and c_goal
            pt_path = os.path.join(self.out_dir, 'plans', 'c_min_%s_%d_epoch_%d.pt' % (metric, i, epoch))
            if os.path.exists(pt_path):
                c_start, c_goal, est_start_obs, est_goal_obs = torch.load(pt_path)
            else:
                _, c_start, _, est_start_obs = self.closest_code(start_obs,
                                                                 400,
                                                                 False,
                                                                 metric, 1)
                _, _, c_goal, est_goal_obs = self.closest_code(goal_obs,
                                                               400,
                                                               True,
                                                               metric, 1)
                # _, c_start, _, est_start_obs = self.closest_code(start_obs,
                #                                                  self.traj_eval_copies,
                #                                                  False,
                #                                                  metric, 0)
                # _, _, c_goal, est_goal_obs = self.closest_code(goal_obs,
                #                                                self.traj_eval_copies,
                #                                                True,
                #                                                metric, 0)
                torch.save([c_start, c_goal, est_start_obs, est_goal_obs], pt_path)

            # Plan using c_start and c_goal.
            rollout = self.planner(c_start.repeat(self.traj_eval_copies, 1),
                                   c_goal.repeat(self.traj_eval_copies, 1),
                                   start_obs=start_obs,
                                   goal_obs=goal_obs)

            # Insert closest start and goal.
            rollout.insert(0, est_start_obs.repeat(self.traj_eval_copies, 1, 1, 1))
            rollout.append(est_goal_obs.repeat(self.traj_eval_copies, 1, 1, 1))

            # Insert real start and goal.
            rollout.insert(0, start_obs.repeat(self.traj_eval_copies, 1, 1, 1))
            rollout.append(goal_obs.repeat(self.traj_eval_copies, 1, 1, 1))

            rollout_best_k, confidences = self.get_best_k(rollout, keep_best)
            rollout_data = torch.stack(rollout_best_k, dim=0)

            masks = - np.ones((rollout_data.size()[0], keep_best, self.channel_dim, 64, 64),
                              dtype=np.float32)
            write_number_on_images(masks, confidences)

            # save_image(torch.max(rollout_data, from_numpy_to_var(masks)).view(-1, self.channel_dim, 64, 64).data,
            #            os.path.join(self.out_dir, 'plans', '%s_min_%s_%d_epoch_%d.png'
            #                         % (self.planner.__name__, metric, i, epoch)),
            #            nrow=keep_best,
            #            normalize=True)

            pd = torch.max(rollout_data, from_numpy_to_var(masks)).t().contiguous().view(-1, self.channel_dim, 64, 64)

            save_image(pd.data,
                       os.path.join(self.out_dir, 'plans', '%s_min_%s_%d_epoch_%d.png'
                                    % (self.planner.__name__, metric, i, epoch)),
                       nrow=int(pd.size()[0]/keep_best),
                       normalize=True)

    def get_best_k(self, rollout, keep_best=10):
        """
        Evaluate confidence using discriminator.
        :param rollout: (list) n x (torch) channel size x W x H
        :param keep_best: get the best keep_best scores.
        :return: rollout list size n x (torch) keep_best x channel size x W x H,
                 confidence np size n-1 x keep_best
        """
        confidences = [self.discriminator_function(rollout[i], rollout[i + 1]).reshape(-1) for i in
                       range(len(rollout) - 1)]
        np_confidences = np.array(confidences)
        # take minimum confidence along trajectory
        min_confidences = np.mean(np_confidences, axis=0)
        # sort according to confidence
        sort_ind = np.argsort(min_confidences, axis=0)
        rollout = [r[sort_ind[-keep_best:]] for r in rollout]
        #confidences = [c[sort_ind[-keep_best:]] for c in confidences]
        np_confidences = np.concatenate([np_confidences[:, sort_ind[-keep_best:]],
                                        np.zeros((1, keep_best))], 0)
        return rollout, np_confidences

    def closest_code(self, obs, n_trials, use_second, metric, regress_bs, verbose=True):
        """
        Get the code that generate the closest distance.
        :param obs: 1 x channel_dim x img_W x img_H
        :param n_trials: number of copies to search
        :param use_second: bool, to measure distance using the second image
        :param metric: str, choose either l2 or D to measure distance
        :param regress_bs: int, regression batch size when 0 do just sampling.
        :return: the best noise and codes
        """
        if metric == 'L2':
            f = lambda x, y: ((x - y) ** 2).view(n_trials, -1).sum(1)
        elif metric == 'load_D':
            f = lambda x, y: - self.load_D(x, y).view(-1) + ((x - y) ** 2).view(n_trials, -1).sum(1)/10
        else:
            assert metric == 'D'
            # turned max into min using minus.
            f = lambda x, y: - self.D(x, y).view(-1)

        if regress_bs:
            z_var = Variable(0.1*torch.randn(n_trials, self.rand_z_dim).cuda(), requires_grad=True)
            c_var = Variable(0.1*torch.randn(n_trials, self.c_dim).cuda(), requires_grad=True)
            # c_var = Variable(self.Q.forward_soft(self.FE(obs.repeat(n_trials, 1, 1, 1))).data, requires_grad=True)
            optimizer = optim.Adam([c_var, z_var], lr=1e-2)
            n_iters = 1000
            for i in range(n_iters):
                optimizer.zero_grad()
                if self.planner == self.astar_plan:
                    c = F.tanh(c_var.repeat(regress_bs, 1))
                else:
                    c = c_var.repeat(regress_bs, 1)
                _z = z_var.repeat(regress_bs, 1)

                c_next = self.T(c)
                o, o_next = self.G(_z, c, c_next)

                if use_second:
                    out = o_next
                else:
                    out = o

                dist = f(obs.repeat(n_trials*regress_bs, 1, 1, 1), out).sum(0)/regress_bs
                if i % 100 == 0:
                    print("\t Closest code (%d/%d): %.3f" % (i, n_iters, dist))
                dist.backward()
                optimizer.step()

            _z = z_var.detach()
            if self.planner == self.astar_plan:
                c = F.tanh(c_var.detach())
            else:
                c = c_var.detach()
        else:
            _z = Variable(torch.randn(n_trials, self.rand_z_dim)).cuda()
            c = self.Q.forward_soft(self.FE(obs)).repeat(n_trials, 1)

        # Select best c and c_next from different initializations.
        if self.planner == self.astar_plan:
            c_next = torch.clamp(self.T(c), -1 + 1e-3, 1 - 1e-3)
        else:
            c_next = self.T(c)
        o, o_next = self.G(_z, c, c_next)
        if use_second:
            out = o_next
        else:
            out = o

        dist = f(obs.repeat(n_trials, 1, 1, 1), out)
        min_dist, min_idx = dist.min(0)
        if verbose:
            # import ipdb; ipdb.set_trace()
            print("\t best_c: %s" % print_array(c[min_idx.data[0]].data))
            print("\t best_c_next: %s" % print_array(c_next[min_idx.data[0]].data))
            print('\t %s measure: %.3f' % (metric, min_dist))
        return _z[min_idx].detach(), c[min_idx].detach(), c_next[min_idx].detach(), out[min_idx].detach()

    def simple_plan(self, c_start, c_goal, verbose=True, **kwargs):
        """
        Generate a plan in observation space given start and goal states.
        :param c_start: bs x c_dim
        :param c_goal: bs x c_dim
        :return: rollout: horizon x bs x channel_dim x img_W x img_H
        """
        with torch.no_grad():
            rollout = []
            half_z = int(self.rand_z_dim / 2)
            _z = Variable(torch.randn(c_start.size()[0], self.rand_z_dim)).cuda()
            for t in range(self.planning_horizon):
                c = c_start + (c_goal - c_start) * t/ self.planning_horizon
                c_next = c_start + (c_goal - c_start) * (t + 1) / self.planning_horizon
                # _z = Variable(torch.randn(c.size()[0], self.rand_z_dim)).cuda()

                _cur_img, _next_img = self.G(_z, c, c_next)
                if t == 0:
                    rollout.append(_cur_img)
                if self.G.gtype == 70:
                    # Gtype 70 only
                    next_img = self.G.conditional_g(rollout[-1], torch.cat([_z[:, half_z:], c_next], 1))
                elif self.G.gtype == 71:
                    next_img = self.G.conditional_g(rollout[-1], torch.cat([_z[:, half_z:], c_next - c], 1))
                elif self.G.gtype == 6:
                    next_img = _next_img
                elif self.G.gtype == 92:
                    if t > 0:
                        assert torch.equal(_cur_img, rollout[-1])
                    next_img = _next_img
                else:
                    raise AssertionError("The code should never have got here.")

                rollout.append(next_img)
                if verbose:
                    # import ipdb; ipdb.set_trace()
                    print("\t c_%d: %s" % (t, print_array(c[0].data)))
                    # print("\t Transition var: %s" % print_array(self.T.get_var(c_start[0, None]).data[0]))
                    # print("\t Direction: %s" % print_array((c_goal-c_start).data[0]/self.planning_horizon))
        return rollout

    def astar_plan(self, c_start, c_goal, verbose=True, **kwargs):
        with torch.no_grad():
            rollout = []
            half_z = int(self.rand_z_dim / 2)
            # _z = Variable(torch.randn(c_start.size()[0], self.rand_z_dim)).cuda()
            bs = c_start.size()[0]
            traj = plan_traj_astar(
                kwargs['start_obs'],
                kwargs['goal_obs'],
                start_state=c_start[0].data.cpu().numpy(),
                goal_state=c_goal[0].data.cpu().numpy(),
                transition_function = self.continuous_transition_function,
                preprocess_function = self.preprocess_function,
                discriminator_function = self.discriminator_function_np,
                generator_function = self.conditional_generator_function)

            for t, disc in enumerate(traj[:-1]):
                state = undiscretize(disc.state, self.discretization_bins, self.P.unif_range)
                state_next = undiscretize(traj[t+1].state, self.discretization_bins, self.P.unif_range)
                c = from_numpy_to_var(state).repeat(bs, 1)
                c_next = from_numpy_to_var(state_next).repeat(bs, 1)
                _z = Variable(torch.randn(c.size()[0], self.rand_z_dim)).cuda()

                _cur_img, _next_img = self.G(_z, c, c_next)
                if t == 0:
                    rollout.append(_cur_img)
                if self.G.gtype == 70:
                    # Gtype 70 only
                    next_img = self.G.conditional_g(rollout[-1], torch.cat([_z[:, half_z:], c_next], 1))
                elif self.G.gtype == 71:
                    next_img = self.G.conditional_g(rollout[-1], torch.cat([_z[:, half_z:], c_next - c], 1))
                elif self.G.gtype == 6:
                    next_img = _next_img
                elif self.G.gtype == 92:
                    if t > 0:
                        assert torch.equal(_cur_img, rollout[-1])
                    next_img = _next_img
                else:
                    raise AssertionError("The code should never have got here.")

                rollout.append(next_img)
                if verbose:
                    # import ipdb; ipdb.set_trace()
                    print("\t c_%d: %s" % (t, print_array(c[0].data)))
        return rollout
