from utils import *

class SelfAttentionMask_Network(nn.Module):
    def __init__(self, env_id,device = 'cpu',mode = None):
        super(SelfAttentionMask_Network,self).__init__()
        env = gym.make(env_id)
        self.ni = env.observation_space.shape[0]
        self.nh = 128
        self.no = env.action_space.n
        self.device = device
        self.to(self.device)
        self.convert = lambda x: torch.FloatTensor(x).to(self.device) if type(x)!= torch.tensor else x
        self.name = "SelfAttentionMask"
        if mode == "si":
            LogStd = SILogStd
        elif mode == 'sd':
            LogStd = SDLogStd
        self.backbone = nn.Linear(self.ni,self.nh)
        self.mask = nn.ModuleList([Attention(self.nh),
                                   Attention(self.nh)])
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
        self.divergence_term = lambda x,y: -F.cosine_similarity(x,y)
        
    def actor_forward(self,x):
        x_s = F.leaky_relu(self.actor[0](x))
        x = self.actor[1][0](x_s,x_s,x_s)
        return self.actor[-1](x)
        
    def critic_forward(self,x):
        x_s = F.leaky_relu(self.critic[0](x))
        x = self.critic[1][1](x_s,x_s,x_s)
        return self.critic[-1](x)
    
    def get_masked_repr(self,x):
        x= torch.tensor(x).float()
        x_s = F.leaky_relu(self.backbone(x)).detach()
        x_pi = self.mask[0](x_s,x_s,x_s)
        x_v = self.mask[1](x_s,x_s,x_s)
        return torch.stack([x_pi,x_v])

class SharedMask_Network(nn.Module):
    def __init__(self, env_id,device = 'cpu',mode = None):
        super(SharedMask_Network,self).__init__()
        env = gym.make(env_id)
        self.ni = env.observation_space.shape[0]
        self.nh = 128
        self.no = env.action_space.n
        self.device = device
        self.to(self.device)
        self.convert = lambda x: torch.FloatTensor(x).to(self.device) if type(x)!= torch.tensor else x
        self.name = "SharedMask"
        self.backbone = nn.Linear(self.ni,self.nh)
        self.mask = Attention(self.nh)
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
        self.divergence_term = lambda x,y: -F.cosine_similarity(x,y)
        
    def actor_forward(self,x):
        x_s = F.leaky_relu(self.actor[0](x))
        x = self.actor[1](x_s,x_s,x_s)
        return self.actor[-1](x)
        
    def critic_forward(self,x):
        x_s = F.leaky_relu(self.critic[0](x))
        x = self.critic[1](x_s,x_s,x_s,True)
        return self.critic[-1](x)
    
    def get_masked_repr(self,x):
        x= torch.tensor(x).float()
        x_s = F.leaky_relu(self.backbone(x)).detach()
        x_pi = self.mask(x_s,x_s,x_s)
        x_v = self.mask(x_s,x_s,x_s,True)
        return torch.stack([x_pi,x_v])
    
class LatentQueryMask_Network(nn.Module):
    def __init__(self, env_id,device = 'cpu',mode=None):
        super(LatentQueryMask_Network,self).__init__()
        env = gym.make(env_id)
        self.ni = env.observation_space.shape[0]
        self.nh = 128
        self.no = env.action_space.n
        self.device = device
        self.to(self.device)
        self.convert = lambda x: torch.FloatTensor(x).to(self.device) if type(x)!= torch.tensor else x
        self.name = "LatentQueryMask"
        self.backbone = nn.Linear(self.ni,self.nh)
        self.latent_query = LatentQuery(self.nh)
        self.mask = nn.ModuleList([Attention(self.nh),
                                   Attention(self.nh),
                                   self.latent_query])
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
        self.divergence_term = lambda x,y: -F.cosine_similarity(x,y)
                
    def actor_forward(self,x):
        x_s = F.leaky_relu(self.actor[0](x))
        x = self.actor[1][0](x_s,x_s,self.latent_query(x_s))
        return self.actor[-1](x)
        
    def critic_forward(self,x):
        x_s = F.leaky_relu(self.critic[0](x))
        x = self.critic[1][1](x_s,x_s,self.latent_query(x_s))
        return self.critic[-1](x)
    
    def get_masked_repr(self,x):
        x= torch.tensor(x).float()
        x_s = F.leaky_relu(self.backbone(x)).detach()
        x_pi = self.mask[0](x_s,x_s,self.latent_query(x_s))
        x_v = self.mask[1](x_s,x_s,self.latent_query(x_s),True)
        return torch.stack([x_pi,x_v])