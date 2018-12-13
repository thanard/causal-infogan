# Learning Plannable Representation with Causal InfoGAN

[Paper](https://arxiv.org/abs/1807.09341)

Causal InfoGAN (CIGAN) takes sequential observations as input, defining what is possible in the system, e.g., a piece of rope can move slightly but cannot be broken in half. It learns a latent state transition P(s'|s) that is simple to plan with. A feasible plan in the latent space generates a visual plan to bring the start configuration to the goal configuration.

In this repo, we provide the code to imagine rope manipulation from random exploration data. The latent space is continuous with locally Gaussian state transition. Therefore, linear interpolation can generate a feasible plan. Note that as described in the paper [1] one can use different representations and planning algorithms in the latent space.

## Setup
**1) Datasets**
- Download rope dataset [2] ([rope_full](https://drive.google.com/uc?export=download&confirm=ypZ7&id=10xovkLQ09BDvhtpD_nqXWFX-rlNzMVl9))
- Download test start and goal images ([seq_data_2](https://drive.google.com/file/d/1n8Yw1fQ2tzvWMWYvTpzzNANWgUVI5Vsl/view?usp=sharing))
- Download the parameters of a fully connected network that is trained to extract the rope from the background ([FCN_mse](https://drive.google.com/file/d/1VGV_QYh24mQH-XVnJYuXRnijWPdu2ojD/view?usp=sharing))

**2) Install the python environment**
- Create a python environment and install dependencies: `conda env create -f tf14.yml`
- Activate the environment: `source activate tf14`

**3) Run the training**
- Run `python main.py -learn_var -seed 2`

![cigan_result](https://github.com/thanard/causal-infogan/blob/master/causal_infogan.png)

## Notes
1) The training is configured to run on a GPU. One can run on a CPU by removing `.cuda()`.
2) We found that some random seeds can collapse early. We are curious to see how techniques in improving GAN stability and mode collapsing be applied here.
   > Because we search for the closest L2 distance on the image space to embed the start and goal images using the generator, more diversity in generation will improve the embeddings of starts and goals.

## References
[1] Thanard Kurutach, Aviv Tamar, Ge Yang, Stuart J. Russell, and Pieter Abbeel. "Learning plannable representations with causal infogan." In Advances in Neural Information Processing Systems, pp. 8746-8757. 2018.

[2] Ashvin Nair, Dian Chen, Pulkit Agrawal, Phillip Isola, Pieter Abbeel, Jitendra Malik, and Sergey Levine. "Combining self-supervised learning and imitation for vision-based rope manipulation." In Robotics and Automation (ICRA), 2017 IEEE International Conference on, pp. 2146-2153. IEEE, 2017.
