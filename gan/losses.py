import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    
    loss = None

    device = logits_real.device
    
    loss1 = bce_loss(logits_real, torch.ones_like(logits_real, dtype=torch.float, device=device), reduction='mean')
    loss2 = bce_loss(logits_fake, torch.zeros_like(logits_fake, dtype=torch.float, device=device), reduction='mean')

    loss = loss1 + loss2
        
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    
    loss = None
    device = logits_fake.device
    loss = bce_loss(logits_fake, torch.ones_like(logits_fake, dtype=torch.float, device=device), reduction='mean')

    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    device = scores_real.device
    #LSGAN discriminator always tries to push scores towards 1 for real images & towards 0 for fake images.
    mse_loss = torch.nn.MSELoss(reduction='mean')

    loss1 = mse_loss(scores_real, torch.ones_like(scores_real, dtype=torch.float, device=device))
    loss2 = mse_loss(scores_fake, torch.zeros_like(scores_fake, dtype=torch.float, device=device))

    loss = 0.5 * (loss1 + loss2)
    
    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None

    device = scores_fake.device
    
    #LSGAN generator always tries to push fake scores towards 1, so trying to fool discriminator.
    mse_loss = torch.nn.MSELoss(reduction='mean')

    loss =  mse_loss(scores_fake, torch.ones_like(scores_fake, dtype=torch.float, device=device))
    
    return 0.5 * loss
