from utils import *

class MLPMixer_Network(nn.Module):
    def __init__(self, env_id,device = 'cpu',mode=None):
        super(MLPMixer_Network,self).__init__()
        env = gym.make(env_id)
        self.ni = env.observation_space.shape[0]
        self.nh = 128
        self.no = env.action_space.n
        self.device = device
        self.to(self.device)
        self.convert = lambda x: torch.FloatTensor(x).to(self.device) if type(x)!= torch.tensor else x
        self.name = "TMLPMixer"
        self.mixer = nn.Sequential(MixerMLP(self.nh),
                                   MixerMLP(self.nh),
                                   MixerMLP(self.nh))
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
        x = torch.stack([x_pi,x_v])
        x = self.actor[1](x)
        return self.actor[-1](x[0])
        
    def critic_forward(self,x):
        x_pi = F.leaky_relu(self.actor[0](x)).detach()
        x_v = F.leaky_relu(self.critic[0](x))
        x = torch.stack([x_pi,x_v])
        x = self.critic[1](x)
        return self.critic[-1](x[1])
    
    def get_mixed_repr(self,x):
        x= torch.tensor(x).float()
        x_pi = F.leaky_relu(self.actor[0](x)).detach()
        x_v = F.leaky_relu(self.critic[0](x)).detach()
        x = torch.stack([x_pi,x_v])
        x = self.mixer(x)
        return x
    
class ConvMixer_Network(nn.Module):
    def __init__(self, env_id,device = 'cpu',mode="si"):
        super(ConvMixer_Network,self).__init__()
        env = gym.make(env_id)
        self.ni = env.observation_space.shape[0]
        self.nh = 128
        self.no = env.action_space.n
        self.device = device
        self.to(self.device)
        self.convert = lambda x: torch.FloatTensor(x).to(self.device) if type(x)!= torch.tensor else x
        self.name = "ConvMixer"
        self.mixer = nn.Sequential(conv(),
                                   conv(),
                                   conv())
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
        self.similarity_term = lambda x,y: -F.cosine_similarity(x,y)
        self.divergence_term = None
        self.get_masked_repr=None
        
    def actor_forward(self,x):
        x_pi = F.leaky_relu(self.actor[0](x))
        x_v = F.leaky_relu(self.critic[0](x)).detach()
        x = torch.stack([x_pi,x_v]).unsqueeze(0)
        x = self.actor[1](x).squeeze()
        return self.actor[-1](x[0])
        
    def critic_forward(self,x):
        x_pi = F.leaky_relu(self.actor[0](x)).detach()
        x_v = F.leaky_relu(self.critic[0](x))
        x = torch.stack([x_pi,x_v]).unsqueeze(0)
        x = self.critic[1](x).squeeze()
        return self.critic[-1](x[1])
    
    def get_mixed_repr(self,x):
        x= torch.tensor(x).float()
        x_pi = F.leaky_relu(self.actor[0](x)).detach()
        x_v = F.leaky_relu(self.critic[0](x)).detach()
        x = torch.stack([x_pi,x_v]).unsqueeze(0)
        x = self.mixer(x).squeeze()
        return torch.stack((x[0],x[1]))

class AttentionMixer_Network(nn.Module):
    def __init__(self, env_id,device = 'cpu',mode=None):
        super(AttentionMixer_Network,self).__init__()
        env = gym.make(env_id)
        self.ni = env.observation_space.shape[0]
        self.nh = 128
        self.no = env.action_space.n
        self.device = device
        self.to(self.device)
        self.convert = lambda x: torch.FloatTensor(x).to(self.device) if type(x)!= torch.tensor else x
        self.name = "AttentionMixer"
        self.mixer = nn.ModuleList([Attention(self.nh),
                                    Attention(self.nh)])
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
        self.similarity_term = lambda x,y: -F.cosine_similarity(x,y)
        self.divergence_term = None
        self.get_masked_repr=None
        
    def actor_forward(self,x):
        x_pi = F.leaky_relu(self.actor[0](x))
        x_v = F.leaky_relu(self.critic[0](x)).detach()
        x = self.actor[1][0](x_v,x_pi,x_pi)
        return self.actor[-1](x)
        
    def critic_forward(self,x):
        x_pi = F.leaky_relu(self.actor[0](x)).detach()
        x_v = F.leaky_relu(self.critic[0](x))
        x = self.critic[1][1](x_pi,x_v,x_v)
        return self.critic[-1](x)
    
    def get_mixed_repr(self,x):
        x= torch.tensor(x).float()
        x_pi = F.leaky_relu(self.actor[0](x)).detach()
        x_v = F.leaky_relu(self.critic[0](x)).detach()
        x = torch.stack([x_pi,x_v])
        x_pi = self.mixer[0](x_v,x_pi,x_pi)
        x_v = self.mixer[1](x_pi,x_v,x_v)
        return torch.stack([x_pi,x_v])