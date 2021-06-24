nh = 128

class MLPMix_Network(nn.Module):
    def __init__(self, env_id,device = 'cpu',mode=None):
        super(MLPMix_Network,self).__init__()
        env = gym.make(env_id)
        self.ni = env.observation_space.shape[0]
        self.nh = nh
        self.no = env.action_space.n
        self.device = device
        self.to(self.device)
        self.convert = lambda x: torch.FloatTensor(x).to(self.device) if type(x)!= torch.tensor else x
        self.name = "MLPMixer"
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
        x_v = F.leaky_relu(self.critic[0](x)).detach()
        x = torch.cat([x_pi,x_v])
        x = self.actor[1][0](x)
        return self.actor[-1](x)
        
    def critic_forward(self,x):
        x_pi = F.leaky_relu(self.actor[0](x)).detach()
        x_v = F.leaky_relu(self.critic[0](x))
        x = torch.cat([x_pi,x_v])
        x = self.critic[1][1](x)
        return self.critic[-1](x)
    
    def get_mixed_repr(self,x):
        x= torch.tensor(x).float()
        x_pi = F.leaky_relu(self.actor[0](x)).detach()
        x_v = F.leaky_relu(self.critic[0](x)).detach()
        x = torch.cat([x_pi,x_v])
        x_pi = self.mixer[0](x)
        x_v = self.mixer[1](x)
        return torch.stack((x_pi,x_v))
    
class rMLPMix_Network(MLPMix_Network):
    def __init__(self, env_id,device = 'cpu',mode=None):
        super(rMLPMix_Network,self).__init__(env_id,device = 'cpu',mode=None)
        
    def actor_forward(self,x):
        x_pi = F.leaky_relu(self.actor[0](x))
        x_v = F.leaky_relu(self.critic[0](x)).detach()
        x = torch.cat([x_pi,x_v])
        x = self.actor[1][0](x)
        return self.actor[-1](x+x_pi)
        
    def critic_forward(self,x):
        x_pi = F.leaky_relu(self.actor[0](x)).detach()
        x_v = F.leaky_relu(self.critic[0](x))
        x = torch.cat([x_pi,x_v])
        x = self.critic[1][1](x)
        return self.critic[-1](x+x_v)
    
class dMLPMix_Network(MLPMix_Network):
    def __init__(self, env_id,device = 'cpu',mode=None):
        super(dMLPMix_Network,self).__init__(env_id,device = 'cpu',mode=None)
        self.critic = nn.ModuleList([nn.Sequential(nn.Linear(self.ni,self.nh),
                                                   nn.LeakyReLU(),
                                                   nn.Linear(self.nh,self.nh)),
                                     self.mixer,
                                     nn.Linear(self.nh*2,1)])
        self.actor = nn.ModuleList([nn.Sequential(nn.Linear(self.ni,self.nh),
                                                  nn.LeakyReLU(),
                                                  nn.Linear(self.nh,self.nh)),
                                    self.mixer,
                                    nn.Sequential(nn.Linear(self.nh*2,self.no),
                                                  nn.Softmax(0))])
        set_seed()
        self.apply(init_weights)
        self.actor.forward = self.actor_forward
        self.critic.forward = self.critic_forward
        
    def actor_forward(self,x):
        x_pi = F.leaky_relu(self.actor[0](x))
        x_v = F.leaky_relu(self.critic[0](x)).detach()
        x = torch.cat([x_pi,x_v])
        x = self.actor[1][0](x)
        return self.actor[-1](torch.cat([x,x_pi]))
        
    def critic_forward(self,x):
        x_pi = F.leaky_relu(self.actor[0](x)).detach()
        x_v = F.leaky_relu(self.critic[0](x))
        x = torch.cat([x_pi,x_v])
        x = self.critic[1][1](x)
        return self.critic[-1](torch.cat([x,x_v]))

class MLPMask_Network(nn.Module):
    def __init__(self, env_id,device = 'cpu',mode=None):
        super(MLPMask_Network,self).__init__()
        env = gym.make(env_id)
        self.ni = env.observation_space.shape[0]
        self.nh = 128
        self.no = env.action_space.n
        self.device = device
        self.to(self.device)
        self.convert = lambda x: torch.FloatTensor(x).to(self.device) if type(x)!= torch.tensor else x
        self.name = "MLPMask"
        self.backbone = nn.Linear(self.ni,self.nh)
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
        x = self.actor[1][0](x_s)
        return self.actor[-1](x)
        
    def critic_forward(self,x):
        x_s = F.leaky_relu(self.critic[0](x))
        x = self.critic[1][1](x_s)
        return self.critic[-1](x)
    
    def get_masked_repr(self,x):
        x= torch.tensor(x).float()
        x_s = F.leaky_relu(self.backbone(x)).detach()
        x_pi = self.mask[0](x_s)
        x_v = self.mask[1](x_s)
        return torch.stack([x_pi,x_v])
    
class rMLPMask_Network(MLPMask_Network):
    def __init__(self, env_id,device = 'cpu',mode=None):
        super(rMLPMask_Network,self).__init__(env_id,device = 'cpu',mode=None)
        
    def actor_forward(self,x):
        x_s = F.leaky_relu(self.actor[0](x))
        x = self.actor[1][0](x_s)
        return self.actor[-1](x+x_s)
        
    def critic_forward(self,x):
        x_s = F.leaky_relu(self.critic[0](x))
        x = self.critic[1][1](x_s)
        return self.critic[-1](x+x_s)
    
class dMLPMask_Network(MLPMask_Network):
    def __init__(self, env_id,device = 'cpu',mode=None):
        super(dMLPMask_Network,self).__init__(env_id,device = 'cpu',mode=None)
        self.critic = nn.ModuleList([self.backbone,
                                     self.mask,
                                     nn.Sequential(nn.Linear(self.nh*2,self.nh),
                                                   nn.LeakyReLU(),
                                                   nn.Linear(self.nh,1))])
        self.actor = nn.ModuleList([self.backbone,
                                    self.mask,
                                    nn.Sequential(nn.Linear(self.nh*2,self.nh),
                                                  nn.LeakyReLU(),
                                                  nn.Linear(self.nh,self.no),
                                                  nn.Softmax(0))])
        set_seed()
        self.apply(init_weights)
        self.actor.forward = self.actor_forward
        self.critic.forward = self.critic_forward
        
    def actor_forward(self,x):
        x_s = F.leaky_relu(self.actor[0](x))
        x = self.actor[1][0](x_s)
        return self.actor[-1](torch.cat([x,x_s]))
        
    def critic_forward(self,x):
        x_s = F.leaky_relu(self.critic[0](x))
        x = self.critic[1][1](x_s)
        return self.critic[-1](torch.cat([x,x_s]))