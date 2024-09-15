import torch

class DeepDenoisingProbModel():

    def __init__(self, num_steps, beta_min, beta_max, device='cpu'):

        self.num_steps = num_steps
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.beta = torch.linspace(beta_min, beta_max, num_steps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        self.alpha_bar_sqrt = torch.sqrt(self.alpha_bar)
        self.one_minus_alpha_bar_sqrt = torch.sqrt(1-self.alpha_bar)


    def fwd_process(self, original_img, timestep, noise=None):

        
        ''' Takes in an image and adds noise to it
            image and noise are of BxCxHxW
            timestep is a 1-D tensor of size B
        '''
        original_img_shape = original_img.shape
        batch_size = original_img_shape[0]

        alpha_bar_sqrt = self.alpha_bar_sqrt[timestep].reshape(batch_size)
        one_minus_alpha_bar_sqrt = self.one_minus_alpha_bar_sqrt[timestep].reshape(batch_size)
        
        # for _ in range(len(original_img_shape)-1):

        #     alpha_bar_sqrt = alpha_bar_sqrt.unsqueeze(-1)
        #     one_minus_alpha_bar_sqrt = one_minus_alpha_bar_sqrt.unsqueeze(-1)

        alpha_bar_sqrt = alpha_bar_sqrt.reshape(-1,1,1,1)
        one_minus_alpha_bar_sqrt = one_minus_alpha_bar_sqrt.reshape(-1,1,1,1)
        
        eps = torch.randn_like(original_img) if noise is None else noise

        noisy_img = alpha_bar_sqrt*original_img + one_minus_alpha_bar_sqrt*eps

        return noisy_img
        

    def sample_previous_step(self, X_t, noise_pred, timestep):

        ''' Returns image of previous timestep 
            based on the mean and variance & reparameterization trick
        '''

        variance = ((1-self.alpha_bar[timestep-1])/(1-self.alpha_bar[timestep])) * self.beta[timestep]
        std_dev = variance ** 0.5
        mean = (1/self.alpha_bar_sqrt[timestep]) * (X_t - (((self.beta[timestep])/(self.one_minus_alpha_bar_sqrt[timestep]))*noise_pred))

        if timestep == 0:
            noise = 0
        else:
            _eps = torch.rand_like(X_t)
            noise = std_dev * _eps
        
        X_t_minus_one = mean + noise

        return X_t_minus_one
    

if __name__ == '__main__':

    # experimenting here rough work ignore finally

    model = DeepDenoisingProbModel(10, 0.1, 1)

    og_img = torch.randn(10,3,28,28)
    timestep = torch.full((10,),3)

    fwd_eqn = model.fwd_process(og_img, timestep)
