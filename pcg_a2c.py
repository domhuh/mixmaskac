from utils import *

class PCG_A2C(nn.Module):
    def __init__(self,network,no,
                 mix_funct=None, mask_funct=None,
                 similarity_term=None, divergence_term=None,
                 device='cpu'):
        super(PCG_A2C,self).__init__()
        self.actor = network.actor
        self.critic = network.critic
        self.to(device)
        self.no=no
        self.scores = []
        self.mix_funct = mix_funct
        self.mask_funct = mask_funct
        self.similarity_term=similarity_term
        self.divergence_term=divergence_term
        self.shared_parameters = list(network.mix.parameters())
        #self.shared_parameters = list(network.mask.parameters())
        self.separated_parameters = list(self.actor[0].parameters())+list(self.actor[2:].parameters())+\
                                    list(self.critic[0].parameters())+list(self.critic[2:].parameters())

    def forward(self,obs,epilson=0.1):
        obs= torch.tensor(obs).float().flatten()
        prob = self.actor(obs)
        dist = Categorical(prob)
        if random.random()<epilson:
            action = torch.tensor([self.env.action_space.sample()])
        else:
            action = dist.sample()
        value = self.critic(obs)
        return action.item(), dist.log_prob(action), value, dist.entropy()
    
    def act(self,obs):
        obs= torch.tensor(obs).float().flatten()
        prob = self.actor(obs)
        return prob.argmax().item()

    def fit(self,env_id, gamma = 0.99, alpha_mix=1e-4, alpha_mask=1e-4,
            tau=0.99, k = 1,lr = 1e-2, n=100, plot=False, epilson_decay=0.5,
            dist=False, term_decay = 0.5, z_score = 2.326): 
        """ MC GAE Updates"""
        ref = defaultdict(dict)
        self.env = gym.make(env_id)
        optimizer= PCGrad(optim.Adam(self.parameters()),self.shared_parameters)
        epilson=1.0
        for i in range(k):
            self.train()
            obs = self.env.reset()
            done = False
            log_probs = []
            rewards = []
            entropies = []
            values = []
            mix = []
            mask = []
            while not done:
                action, log_prob, value, H = self(obs,epilson)
                obs, reward, done, _ = self.env.step(action)
                if self.mix_funct is not None:
                    mix.append(self.mix_funct(obs))
                if self.mask_funct is not None:
                    mask.append(self.mask_funct(obs))
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                entropies.append(H)
                
            values = values + [0.0]
            gae = 0
            returns = deque()
            for step in reversed(range(len(rewards))):
                delta = rewards[step] + gamma * values[step + 1] - values[step]
                gae = delta + gamma * tau * gae
                returns.appendleft(gae + values[step])
            returns = np.array(list(returns))
            
            loss =0
            policy_loss, value_loss = 0,0
            for log_prob, value, R, H in zip(log_probs,values, returns, entropies):
                advantage = R - value.item()
                policy_loss += advantage * -log_prob
                value_loss += F.smooth_l1_loss(torch.FloatTensor([R]).detach(),value)
                loss += + 1e-3*-H
            
            if self.mix_funct is not None:# and i%2:
                mix = torch.stack(mix)
                loss += alpha_mix * self.similarity_term(mix[:,0],mix[:,1]).sum()
                alpha_mix = self.update_alpha(alpha_mix,term_decay)
            
            if self.mask_funct is not None:# and not i%2:
                mask = torch.stack(mask)
                loss -= alpha_mask * self.divergence_term(mask[:,0],mask[:,1]).sum()
                alpha_mask = self.update_alpha(alpha_mask,term_decay)
                
            optimizer.zero_grad()
            optimizer.pc_backward([policy_loss, value_loss, loss])
            optimizer.step()
            
            self.scores.append(run_k(env_id,self,n))
            epilson*=epilson_decay
            if plot: self.plot()
            
    def update_alpha(self,alpha,term_decay=1e-3):
        return alpha * term_decay
        
    def plot(self):
        [mean_scores,min_score,max_score] = np.array(self.scores).T
        clear_output(True)
        plt.figure(figsize=(5, 5))
        plt.plot(mean_scores)
        background = colors.to_rgb(plt.gca().lines[-1].get_color())
        background = [*background,0.1]
        plt.fill_between(np.arange(len(self.scores)),y1=min_score,y2=max_score,
                         color=background)
        plt.show()

def run(env_id,policy,proc_num=0,return_dict=None):
    env = gym.make(env_id)
    crewards = 0
    done=False
    obs = env.reset()
    while not done:
        action = policy.act(obs)
        next_obs, reward, done, _ = env.step(action)
        obs = next_obs
        crewards += reward
    if return_dict is not None: return_dict[proc_num] = crewards
    return crewards

def run_k(env_id,policy,k=1):
    rewards = []
    policy.eval()
    for _ in range(k):
        rewards.append(run(env_id, policy))
    return [np.mean(rewards), np.min(rewards), np.max(rewards)]

def run_k_parallel(env_id,policy,k=1):
    manager = mp.Manager(); return_dict = manager.dict()
    jobs = []
    worker = partial(run,env_id, copy.deepcopy(policy))
    for n in range(k):
        p = mp.Process(target=worker, args=(n,return_dict))
        jobs.append(p); p.start()
    for proc in jobs:
        proc.join()
    return [np.mean(return_dict.values()), np.min(return_dict.values()), np.max(return_dict.values())]

class PCGrad():
    def __init__(self, optimizer, conflict_set):
        self._optim = optimizer
        self.conflict_set = conflict_set
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        return self._optim.step()

    def pc_backward(self, objectives):
        grads, shapes, has_grads, flags = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads, flags)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, flags, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i,flag in zip(pc_grad,flags):
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0 and flag==1:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad, flags = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads, flags

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        grad, shape, has_grad, flags = [], [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                flag = 0
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
                for j in self.conflict_set:
                    if id(p)==id(j):
                        flag=1
                        break
                flags.append(flag)
        return grad, shape, has_grad, flags