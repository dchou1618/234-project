import numpy as np
import torch.nn as nn
import torch.optim as opt
import torch
from torch.utils.data import DataLoader

from rubiks_cube_222_lbl_ppo_convert_multibinary import RubiksCube222EnvLBLPPOB
from skewb_multibinary import SkewbEnvB
from pyraminx_multibinary import PyraminxWoTipsEnv

import pickle

class reward_model():
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1) -> None:
        layers = [nn.Linear(input_dim, hidden_dim), 
                  nn.ReLU()]
        for i in range(num_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.optimizer = opt.Adam(self.network.parameters(), lr = 0.01)
    
    
    def train(self,train_loader,num_epoch = 100):
        loss_fn = nn.MSELoss()
        for epoch in range(num_epoch):
            print("epoch:" + str(epoch))
            for batch in train_loader:
                self.optimizer.zero_grad()
                target = batch[:,144]
                #target = batch[:,180]
                keys = batch[:,:144]
                #keys = batch[:,:180]
                input = torch.squeeze(self.network(keys))
                loss = loss_fn(target, input)
                loss.backward()
                self.optimizer.step()
            print(loss.item())
        
        with open("reward_model_pyr.pickle", "wb") as file:
            pickle.dump(self, file)






if __name__ == '__main__':
   # env = RubiksCube222EnvLBLPPOB() 
    # env = SkewbEnvB()
    env = PyraminxWoTipsEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state, _ = env.reset(options = {"scramble": False})
    truth = dict()
    truth[tuple(state)] = 0

    for i in range(100000):
        env.reset(options = {"scramble": False})
        for j in range(14):
            env.algorithm(env.generate_scramble_num(1))
            env.update_cube_reduced()
            env.update_cube_state()
            ##state = tuple(np.eye(6)[env.convert(env.cube)].flatten().tolist())
            state = tuple(env.convert().tolist()) # skewb/pyraminx
            if state not in truth:
                truth[state] = np.inf
            truth[state] = min(truth[state], j+1)

    with open('truth_pyr_100000.pickle','wb') as file:
        pickle.dump(truth,file)
    
    #with open('truth_pyr_100000.pickle','rb') as file:
        #truth = pickle.load(file)
    
    r_mod = reward_model(144, 32, 1, 2)
    #r_mod = reward_model(180, 32, 1, 2) # skewb
    print(r_mod.network.parameters())

    truth_keys = [list(key) for key in truth.keys()]
    print(len(truth_keys))


    train_tensor = torch.hstack((torch.Tensor(truth_keys).to(device).reshape(-1,144),  torch.Tensor(list(truth.values())).reshape(-1,1)))
    train_loader = DataLoader(train_tensor, batch_size=128, shuffle = True)

    r_mod.train(train_loader)
    







    

   

