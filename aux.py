from utils import *

class AuxMLPMix_Network(nn.Module):
    def __init__(self, env_id,device = 'cpu',mode=None):
        super(AuxMLPMix_Network,self).__init__()
        env = gym.make(env_id)
        self.ni = env.observation_space.shape[0]
        self.nh = 128
        self.no = env.action_space.n
        self.device = device
        self.to(self.device)
        self.convert = lambda x: torch.FloatTensor(x).to(self.device) if type(x)!= torch.tensor else x
        self.name = "MetaMLPMixer"
        self.mixer = nn.ModuleList([nn.Sequential(nn.Linear(self.nh*2,self.nh),
                                                  nn.LeakyReLU()),
                                    nn.Sequential(nn.Linear(self.nh*2,self.nh),
                                                  nn.LeakyReLU())])
        self.critic = nn.ModuleList([nn.Sequential(nn.Linear(self.ni,self.nh),
                                                   nn.LeakyReLU(),
                                                   nn.Linear(self.nh,self.nh)),
                                     self.mixer,
                                     nn.Linear(self.nh,1)])
        self.actor = nn.ModuleList([nn.Sequential(nn.Linear(self.ni,self.nh),
                                                  nn.LeakyReLU(),
                                                  nn.Linear(self.nh,self.nh)),
                                    self.mixer,
                                    nn.Sequential(nn.Linear(self.nh,self.no),
                                                  nn.Softmax(0))])
        set_seed()
        self.apply(init_weights)
        self.actor.forward = self.actor_forward
        self.critic.forward = self.critic_forward
        self.get_masked_repr = None
        self.similarity_term = lambda x,y: -F.cosine_similarity(x,y)
        self.divergence_term = None
        
    def actor_forward(self,x):
        x_pi = F.leaky_relu(self.actor[0](x))
        return self.actor[-1](x_pi)
        
    def critic_forward(self,x):
        x_v = F.leaky_relu(self.critic[0](x))
        return self.critic[-1](x_v)
    
    def get_mixed_repr(self,x):
        x= torch.tensor(x).float()
        x_pi = F.leaky_relu(self.actor[0](x))
        x_v = F.leaky_relu(self.critic[0](x))
        x = torch.cat([x_pi,x_v])
        x_pi = self.mixer[0](x)
        x_v = self.mixer[1](x)
        return torch.stack((x_pi,x_v))


class AuxMLPMask_Network(nn.Module):
    def __init__(self, env_id,device = 'cpu',mode=None):
        super(AuxMLPMask_Network,self).__init__()
        env = gym.make(env_id)
        self.ni = env.observation_space.shape[0]
        self.nh = 128
        self.no = env.action_space.n
        self.device = device
        self.to(self.device)
        self.convert = lambda x: torch.FloatTensor(x).to(self.device) if type(x)!= torch.tensor else x
        self.name = "MLPMask"
        self.backbone = nn.Sequential(nn.Linear(self.ni,self.nh),
                                     nn.LeakyReLU(),
                                     nn.Linear(self.nh,self.nh))
        self.mask = nn.ModuleList([nn.Sequential(nn.Linear(self.nh,self.nh),
                                                 nn.LeakyReLU()),
                                   nn.Sequential(nn.Linear(self.nh,self.nh),
                                                 nn.LeakyReLU())])
        self.critic = nn.ModuleList([self.backbone,
                                     self.mask,
                                     nn.Sequential(nn.Linear(self.nh,self.nh),
                                                   nn.LeakyReLU(),
                                                   nn.Linear(self.nh,1))])
        self.actor = nn.ModuleList([self.backbone,
                                    self.mask,
                                    nn.Sequential(nn.Linear(self.nh,self.nh),
                                                  nn.LeakyReLU(),
                                                  nn.Linear(self.nh,self.no),
                                                  nn.Softmax(0))])
        set_seed()
        self.apply(init_weights)
        self.actor.forward = self.actor_forward
        self.critic.forward = self.critic_forward
        self.get_mixed_repr = None
        self.similarity_term = None
        self.divergence_term = lambda x,y: F.cosine_similarity(x,y)
        
    def actor_forward(self,x):
        x_s = F.leaky_relu(self.actor[0](x))
        return self.actor[-1](x_s)
        
    def critic_forward(self,x):
        x_s = F.leaky_relu(self.critic[0](x))
        return self.critic[-1](x_s)
    
    def get_masked_repr(self,x):
        x= torch.tensor(x).float()
        x_s = F.leaky_relu(self.backbone(x))
        x_pi = self.mask[0](x_s)
        x_v = self.mask[1](x_s)
        return torch.stack([x_pi,x_v])