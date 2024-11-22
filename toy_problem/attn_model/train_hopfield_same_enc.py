from copy import deepcopy
import torch 
import numpy as np
from torch.utils.data import DataLoader
from data_classes import PICalibData, EnsembleTrainLoader, EnsembleValLoader, ValOutput, MemoryData
from typing import Tuple, List, Optional
from conformal_hopfield_batch_same_enc import ConformHopfieldBatchSameEnc
from tqdm import tqdm
import multiprocessing as mp
from utils import plot_part_mhn, check_or_make_folder, plot_part_mhn_many, MeanStdevFilter
import pickle
import time
import sys, os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrainHopfieldSameEnc:

    def __init__(self, params: dict):
        self._params = params
        #self.param = params
        self._epochs = params['mhn_epochs']
        self._lr = params['mhn_lr']
        self._l2_reg = params['mhn_l2_reg'] # TODO: Change it to mhn_l2_reg 
        self._models = ConformHopfieldBatchSameEnc(ctx_enc_in=params['input_dim'],\
                                                ctx_enc_out=params['ctx_enc_out'],\
                                                params=self._params)
                                                         
        self._optims = torch.optim.AdamW(self._models.parameters(),\
                                       lr=self._lr,\
                                        weight_decay=self._l2_reg)
        self._current_best_wghts = self._models.state_dict()
        self._current_best_losses = float('inf')
        self.num_mhn_models = params['num_mhn_models']
        self._use_base_enc = self._params['use_base_enc']
        self.ctx_out_dim = params['ctx_enc_out']
        self.input_filter: MeanStdevFilter = None # initialize it in train method 
        self.output_filter: MeanStdevFilter = None 

    def train_hopfield(self, calib_data: Tuple[PICalibData,...]):
        """
        Train the Modern Hopfield Network (MHN) for calibration.
        """
        print(f"====================== Training MHN ======================")
        start_time = time.time()
        for epoch in range(self._params['mhn_epochs']):

            #for name, param in self._models.named_parameters():
            #    print(f"{name} and {param.shape}")
            """
            _encoder.linear_stack.0.weight and torch.Size([19, 19])
            _encoder.linear_stack.0.bias and torch.Size([19])
            _hopfield.association_core.q_proj_weight and torch.Size([19, 19])
            _hopfield.association_core.k_proj_weight and torch.Size([19, 19])
            _hopfield.association_core.v_proj_weight and torch.Size([1, 1])
            _hopfield.association_core.in_proj_bias and torch.Size([39])
            _hopfield.norm_stored_pattern.weight and torch.Size([19])
            _hopfield.norm_stored_pattern.bias and torch.Size([19])
            _hopfield.norm_state_pattern.weight and torch.Size([19])
            _hopfield.norm_state_pattern.bias and torch.Size([19])    
            """   
            print(f"====================== Epoch {epoch} ======================")
            if epoch % 5 == 0:
                val_output = self.val_hopfield(calib_data)
                val_loss = val_output[0] # losses for all outputs 

            #pbar = tqdm(self._train_loader, desc=f'Train Epoch {epoch}')    
            self._models.train()
            epoch_losses = []
  
            for _, train_data in enumerate(self._train_loader):

                losses = self._models.forward(train_data, train=True) # train losses of all outputs 
                # TAKE MEAN LOSS OF ALL BATCHES 
                #losses = self._params['norm_loss_wghts'][:self._params['num_mhn_models']]*losses
                loss = torch.mean(losses) 
                self._optims.zero_grad() 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._models.parameters(), max_norm=1.5)
                self._optims.step()
                epoch_losses.append(loss.item())
       
            epoch_avg_loss = np.mean(np.array(epoch_losses))
            val_avg_loss = np.mean(np.array(val_loss))
            print(f"Epoch {epoch}: Train Avg. Loss {epoch_avg_loss} & Val Avg. Loss {val_avg_loss}")
            train = ["Train:"] + [f"A{i}: {loss}" for i, loss in enumerate(losses)]
            print(' '.join(train))   
            val = ["Val:"] + [f"A{i}: {loss}" for i, loss in enumerate(val_loss)]
            print(' '.join(val))             

            #if val_avg_loss < self._current_best_losses:
            self._train_losses_all = ' '.join(train)
            self._val_losses_all = ' '.join(val)
            self._current_best_losses = val_avg_loss
            self._current_best_wghts = self._models.state_dict()

            total_norm = 0
            for p in self._models.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f'Epoch {epoch+1}, Gradient Norm: {total_norm:.4f}')      

        print(f"Best Val Loss during training is: {self._current_best_losses}")
        print(f"Training Models took {time.time()-start_time}s") 
         
        # plot here for now
        save_dir = f"./{self._params['dataset_name']}/mhn_model_shared/{self.num_mhn_models}encHeads_{self._params['mhn_batch_size']}bs_{self._params['ctx_enc_out']}outdim_{self._params['mhn_lr']}lr_{self._params['mhn_l2_reg']}l2_{self._params['mhn_epochs']}epch_{self._params['seq_len']}seqlen_{self._params['num_heads']}heads"
        check_or_make_folder(save_dir + "/val_figs")
        for plot in range(2):
            cat_mu = np.concatenate((val_output[1].y_low[:,plot,:,:,:],\
                                        val_output[1].y_high[:,plot,:,:,:]), axis=1)
            data_dict = {'cat_mu': cat_mu, "gr_tr": val_output[1].y[plot]}
            pickle.dump(data_dict, open(save_dir + '/error_val_data.pkl', 'wb'))                    

        return 

    def val_hopfield(self, calib_data: Tuple[PICalibData]):    
        """
        Validate the Modern Hopfield Network (MHN) after every few epochs.
        """
        #pbar = tqdm(self._val_loader, desc=f'Val Epoch {epoch}')
        self._models.eval()
        # size of final output pred
        wink_scores = [] # list of lists 
        num_heads = self._params['num_heads']

        no_of_episodes = calib_data[1].Y.shape[1] # [models,episodes,seq_len,ctxt_dim]
        alphas = len(self._models._alphas)
        seq_length = calib_data[1].Y.shape[2]
        feat = self._params['num_mhn_models'] 
        val_out = ValOutput(y=np.zeros((no_of_episodes,seq_length,feat)),\
                            y_low=np.zeros((num_heads,no_of_episodes,alphas,seq_length,feat)),\
                            y_high=np.zeros((num_heads,no_of_episodes,alphas,seq_length,feat)))
                            
        with torch.no_grad():
            for batch_idx, val_data in enumerate(self._val_loader): 

                out = self._models.forward(val_data, train=False)
                wink_scores.append(out[0].detach().cpu().numpy())
                start_idx = batch_idx*val_data[0].shape[0] 
                end_idx = (batch_idx+1)*val_data[0].shape[0]
                if self._params['mhn_output'] == 'y':
                    # saving output pred with bounds and ground truth
                    val_out.y[start_idx:end_idx,:,:] = out[1].detach().cpu().numpy()
                    val_out.y_low[:,start_idx:end_idx,:,:,:] = out[2].detach().cpu().numpy()
                    val_out.y_high[:,start_idx:end_idx,:,:,:] = out[3].detach().cpu().numpy() 
                      
        if self._params['mhn_output'] == 'y':
            return np.array(wink_scores).mean(axis=0), val_out
    

    def eval_hopfield(self, calib_data: Tuple[PICalibData,...]):
        """
        Evaluate the hopfield on val data with data in self._models[id]._mem_data
        """
        print(f"**************** Evaluation ****************")
        # encode the X from true train data for evaluation
        for m in range(self.num_mhn_models):
            self._models._mem_data[m] = self.encode_train(calib_data, m)

        if self._params['ood_pred']:
            self.ood_eval()
            return None
        else:        
            no_of_episodes = calib_data[0].Y.shape[1]
            num_heads = self._params['num_heads']
            alphas = len(self._models._alphas)
            seq_length = calib_data[0].Y.shape[2]
            feat = self._params['num_mhn_models'] 
            val_out = ValOutput(y=np.zeros((no_of_episodes,seq_length,feat)),\
                                y_low=np.zeros((num_heads,no_of_episodes,alphas,seq_length,feat)),\
                                y_high=np.zeros((num_heads,no_of_episodes,alphas,seq_length,feat)),)

            pbar = tqdm(self._val_loader, desc=f'Evaluation')
            self._models.eval()

            samples_processed = 0
            samp_to_viz = 300

            with torch.no_grad():
                for batch_idx, val_data in enumerate(pbar): 

                    out = self._models.evaluate(val_data, batch_idx=batch_idx)
                    start_idx = batch_idx*val_data[0].shape[0] 
                    end_idx = (batch_idx+1)*val_data[0].shape[0]
                    # saving output pred with bounds and ground truth
                    val_out.y[start_idx:end_idx,:,:] = out[0].detach().cpu().numpy()
                    val_out.y_low[:,start_idx:end_idx,:,:,:] = out[1].detach().cpu().numpy()
                    val_out.y_high[:,start_idx:end_idx,:,:,:] = out[2].detach().cpu().numpy()   

                    # want to have 32 examples for plotting 
                    samples_processed += val_data[0].shape[0]
                    if samples_processed >= samp_to_viz:
                        break

            all_cat_mu_y = []
            all_gr_tr_y = []

            for plot in range(samp_to_viz):
                cat_mu = np.concatenate((val_out.y_low[:,plot,:,:,:],\
                                            val_out.y_high[:,plot,:,:,:]), axis=1)
                gr_tr = val_out.y[plot] # [seq_len,_which_out]
                
                all_cat_mu_y.append(np.expand_dims(cat_mu, axis=0))
                all_gr_tr_y.append(np.expand_dims(gr_tr, axis=0))  

            all_cat_mu_y = np.concatenate(all_cat_mu_y, axis=0) # [examples,num_heads,no_of_alphas,seq_len,_which_out]
            all_gr_tr_y = np.concatenate(all_gr_tr_y, axis=0) # [examples,seq_len,_which_out]             
                
            return (all_cat_mu_y, all_gr_tr_y)

    def ood_eval(self):

        data_dict = pickle.load(open("./../../toy_data/data_dict_400data_30samp_unimodal_ext_OOD.pkl", 'rb'))   

        mean_x = data_dict['mean_x']
        std_x = data_dict['std_x']
        norm_x = torch.Tensor(data_dict['norm_x']) # add model, feature and batch dim to follow val_data format
        mean_y = data_dict['mean_y']
        std_y = data_dict['std_y']
        norm_y = torch.Tensor(data_dict['norm_y'])

        # repeat for all models 
        norm_x = norm_x.repeat(self.num_mhn_models,1).unsqueeze(0).unsqueeze(-1)
        norm_y = norm_y.repeat(self.num_mhn_models,1).unsqueeze(0).unsqueeze(-1)

        out = self._models.evaluate((norm_x.to(device), norm_y.to(device)), batch_idx=0) 

        # skip saving y, y_low, y_high for now

        return
    
    def encode_train(self, calib_data: Tuple[PICalibData,...], m: int) -> MemoryData:
        """
        Encode the training data for evaluation and fills the memory
        """
        batch_size = calib_data[0].X_norm.shape[1]
        seq_len = calib_data[0].X_norm.shape[2]
        feat = calib_data[0].X_norm.shape[-1]
        
        mem_data = MemoryData(_X_enc_mem=torch.full((batch_size,seq_len,self.ctx_out_dim), fill_value=float('inf')),
                            _Y_mem=torch.full(tuple(calib_data[0].Y_norm[0].shape), fill_value=float('inf')),
                            _X_mem=torch.full((batch_size,seq_len,feat), fill_value=float('inf'))
                        )
        for i in range(calib_data[0].X_norm.shape[1]): #[models,episodes,seq_len,feat]
            
            mem_data._X_mem[i] = calib_data[0].X_norm[m,i,:,:]
            mem_data._X_enc_mem[i] = self._models.heads[str(m)](calib_data[0].X_norm[m,i,:,:].to(device)).cpu()    
                              
            mem_data._Y_mem[i] = calib_data[0].Y_norm[m,i,:,:]
     
        return mem_data

    def all_train_hopfield(self, m_calib_data: Tuple[PICalibData]) -> None:
        """
        Trains separate MHN model for each neural network output
        """
        self.input_filter = self._params['input_filter']
        self.output_filter = self._params['output_filter']

        self._train_loader, self._val_loader = self._data_loaders(m_calib_data)
        self._save_model(m_calib_data, pre_train=True)

        self.train_hopfield(m_calib_data)

        self._save_model(m_calib_data, False)  

        print("ALL MHN MODELS ARE TRAINED AND SAVED")    

        return

    def _save_model(self, calib_data: Tuple[PICalibData,...], pre_train: bool) -> None:
        """
        To save model's state dict for later use
        """
        save_dir = f"./{self._params['dataset_name']}/mhn_model_shared/{self.num_mhn_models}encHeads_{self._params['mhn_batch_size']}bs_{self._params['ctx_enc_out']}outdim_{self._params['mhn_lr']}lr_{self._params['mhn_l2_reg']}l2_{self._params['mhn_epochs']}epch_{self._params['seq_len']}seqlen_{self._params['num_heads']}heads"
        models_state_dict = dict()

        if not pre_train:
            # Saving MHN weights
            models_state_dict = (self._train_losses_all,
                                        self._val_losses_all,
                                            self._current_best_losses,
                                                self._current_best_wghts)

            torch.save(models_state_dict, save_dir + "/mhn_weights.pt")        

        else:
            check_or_make_folder(f"./{self._params['dataset_name']}")
            check_or_make_folder(f"./{self._params['dataset_name']}/mhn_model_shared")
            check_or_make_folder(save_dir)    
            
            print("MHN dataset is saved now!")
            # Saving MHN data 
            data_dict = {"calib_train": calib_data[0],
                        "calib_val": calib_data[1],
                        "input_filter": self.input_filter,
                        "output_filter": self.output_filter
                        }
            pickle.dump(data_dict, open(save_dir + "/mhn_data.pkl", 'wb'))

        return 


    def all_eval_hopfield(self) -> None:
        """
        Evaluate all MHN using val data
        """
        calib_data = self._load_model()
        print("The train points are:", calib_data[0].X_norm.shape)
        print("The val points are:", calib_data[1].X_norm.shape)
        self._train_loader, self._val_loader = self._data_loaders(calib_data)
        #self._val_loader = self._data_loaders(calib_data)

        cat_gr = self.eval_hopfield(calib_data) 
        
        if self._params['ood_pred']:
            return None
        else:
            all_cat_mu_y = cat_gr[0] # [examples,num_heads,no_of_alphas,seq_len,_which_out] 
            all_gr_tr_y = cat_gr[1]  # [examples,seq_len,no_of_outputs]

            # to plot different examples for first feature concatenate along last dim (remove it later)
            all_y = []
            all_gr = []

            save_dir = f"./{self._params['dataset_name']}/mhn_model_shared/{self.num_mhn_models}encHeads_{self._params['mhn_batch_size']}bs_{self._params['ctx_enc_out']}outdim_{self._params['mhn_lr']}lr_{self._params['mhn_l2_reg']}l2_{self._params['mhn_epochs']}epch_{self._params['seq_len']}seqlen_{self._params['num_heads']}heads"
            check_or_make_folder(save_dir + "/eval_figs/")         

            # adding mean of all alphas
            for example in range(all_cat_mu_y.shape[0]):
                mu_pred = []
                for head in range(self._params['num_heads']):
                    mu_pred.append(np.expand_dims(np.concatenate((np.expand_dims(np.mean(all_cat_mu_y[example][head],\
                                                        axis=0), axis=0), all_cat_mu_y[example][head]), axis=0), axis=0))
    
                mu_pred = np.concatenate(mu_pred, axis=0)  # cat along head dim
                
                all_y.append(mu_pred[np.newaxis,:,:,:,:]) # add example dim
                all_gr.append(all_gr_tr_y[example][np.newaxis,:,:])  

            # cat along example dim
            all_y = np.concatenate(all_y, axis=0) # [examples,num_heads,no_of_alphas+1,seq_len,_which_out] mean of all alphas added 
            all_gr = np.concatenate(all_gr, axis=0) # [examples,seq_len,no_of_outputs]

            data_dict = {'cat_mu': all_y, "gr_tr": all_gr}
            pickle.dump(data_dict, open(save_dir +  "/eval_data.pkl", 'wb'))        

        return      


    def _load_model(self) -> Tuple[PICalibData]:
        """
        Loads MHN weights and data
        """
        save_dir = f"./{self._params['dataset_name']}/mhn_model_shared/{self.num_mhn_models}encHeads_{self._params['mhn_batch_size']}bs_{self._params['ctx_enc_out']}outdim_{self._params['mhn_lr']}lr_{self._params['mhn_l2_reg']}l2_{self._params['mhn_epochs']}epch_{self._params['seq_len']}seqlen_{self._params['num_heads']}heads"
        models_state_dict = torch.load(save_dir + "/mhn_weights.pt", map_location=device)

        # Load MHN weights
        print(f"Train: {models_state_dict[0]}")
        print(f"Val: {models_state_dict[1]}")
        print(f"MHN Model Loss is: {models_state_dict[2]}")
        self._models.load_state_dict(models_state_dict[3])

        # Load MHN data 
        # mhn_data_orig contains random in train and seq in val
        data_state_dict = pickle.load(open(save_dir + '/mhn_data.pkl', 'rb'))
        #data_state_dict = pickle.load(open(save_dir + '/mhn_data_TrainVal_seq.pkl', 'rb'))
        calib_data = tuple((data_state_dict['calib_train'],
                           data_state_dict['calib_val']))
        self.input_filter = data_state_dict['input_filter']
        self.output_filter = data_state_dict['output_filter']
        self._models.input_filter = self.input_filter
        self._models.output_filter = self.output_filter
        
        return calib_data
    
    def _data_loaders(self, calib_data: Tuple[PICalibData,...]) -> Tuple[DataLoader, DataLoader]: # type: ignore
            # CAUTION: For self-attention change encode_train method - push train_sim to memory  
        return DataLoader(EnsembleTrainLoader(X=calib_data[0].X_norm, # TRAIN 0 
                                    Y=calib_data[0].Y_norm), 
                                    batch_size=self._params['mhn_batch_size'], shuffle=True),\
                DataLoader(EnsembleValLoader(X=calib_data[1].X_norm, # VAL 1 
                                    Y=calib_data[1].Y_norm),
                                    batch_size=self._params['mhn_batch_size'], shuffle=False)
    


 