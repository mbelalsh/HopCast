import torch 
import numpy as np
from torch.utils.data import DataLoader
from data_classes import PICalibData, TrainLoader, ValLoader, ValOutput, ValOutputError, MemoryData
from typing import Tuple, List, Optional
from conformal_hopfield_batch import ConformHopfieldBatch
from tqdm import tqdm
import multiprocessing as mp
from utils import plot_part_mhn, check_or_make_folder, plot_part_mhn_many, get_positional_encoding
import pickle
import time
import sys, os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrainHopfield:

    def __init__(self, params: dict):
        self._params = params
        #self.param = params
        self._epochs = params['mhn_epochs']
        self._lr = params['mhn_lr']
        self._l2_reg = params['mhn_l2_reg'] # TODO: Change it to mhn_l2_reg 
        self._models = {i: ConformHopfieldBatch(ctx_enc_in=params['ctxt_dim'],\
                                                ctx_enc_out=params['ctxt_dim'],\
                                                params=self._params,
                                                which_out=i)
                                                for i in range(self._params['num_mhn_models'])}
                                                         
        self._optims = {i: torch.optim.AdamW(self._models[i].parameters(),\
                                       lr=self._lr,\
                                        weight_decay=self._l2_reg)
                                        for i in range(params['num_mhn_models'])}
        self._current_best_wghts = {id: model.state_dict() for id, model in self._models.items()}
        self._current_best_losses = {id: float('inf') for id in self._models.keys()}

    def train_hopfield(self, calib_data: Tuple[PICalibData], id: int):
        """
        Train the Modern Hopfield Network (MHN) for calibration.
        """
        print(f"*************** Training MHN {id+1} ***************")
        start_time = time.time()
        for epoch in range(self._params['mhn_epochs']):

            #for name, param in self._models[id].named_parameters():
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
            if epoch % 5 == 0:
                val_output = self.val_hopfield(calib_data, id, epoch)
                val_loss = val_output[0]

            #pbar = tqdm(self._train_loader, desc=f'Train Epoch {epoch}')    
            self._models[id].train()
            epoch_losses = []
  
            for _, train_data in enumerate(self._train_loader):

                loss = self._models[id].forward(train_data, train=True)                
                #loss = self._models[id].batch_forward(train_data)
                # TAKE MEAN LOSS OF ALL BATCHES 
                #loss = self._params['norm_loss_wghts'][self._models[id]._which_out]*loss
                self._optims[id].zero_grad() 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._models[id].parameters(), max_norm=2.0)
                self._optims[id].step()
                epoch_losses.append(loss.item())

            epoch_avg_loss = np.mean(np.array(epoch_losses))
            print(f"Epoch {epoch}: Train Loss {epoch_avg_loss} & Val Loss {val_loss}")

            if val_loss < self._current_best_losses[id]:
                self._current_best_losses[id] = val_loss
                self._current_best_wghts[id] = self._models[id].state_dict()

            total_norm = 0
            for p in self._models[id].parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f'Epoch {epoch+1}, Gradient Norm: {total_norm:.4f}')      

        print(f"Best Val Loss during training is: {self._current_best_losses[id]}")
        print(f"Training Model {id+1} took {time.time()-start_time}s") 

        # plot here for now
        save_dir = f"./{self._params['dataset_name']}/mhn_model_weights/{self._params['past_ts_ctxt']}tsCtxt_{self._params['past_feat_ctxt']}featCtxt_{self._params['past_pred_ctxt']}predCtxt_{self._params['init_cond_ctxt']}initCtxt_{self._params['mhn_lr']}lr_{self._params['mhn_l2_reg']}l2_{self._params['mhn_epochs']}epch_{self._params['num_heads']}heads"
        check_or_make_folder(save_dir + "/val_figs")
        if self._params['mhn_output'] == 'y':
            #np.expand_dims(val_output[1].y_pred[10],axis=0)
            heads = val_output[1].y_low.shape[0]
            for plot in range(3):
                cat_mu = np.concatenate((np.expand_dims(np.expand_dims(val_output[1].y_pred[plot],axis=0),\
                                                         axis=0).repeat(heads,axis=0),\
                                                            val_output[1].y_low[:,plot,:,:,:],\
                                            val_output[1].y_high[:,plot,:,:,:]), axis=1)
                data_dict = {'cat_mu': cat_mu, "gr_tr": val_output[1].y[plot]}
                pickle.dump(data_dict, open(save_dir + '/y_val_data.pkl', 'wb'))
                for _ in range(2):
                    plot_part_mhn(mu_part=cat_mu, ground_truth=val_output[1].y[plot],\
                                        no_of_outputs=1, fig_name=save_dir+"/val_figs/"+f"y_val_{plot}_{self._params['past_ts_ctxt']}tsCtxt_{self._params['past_feat_ctxt']}featCtxt_{self._params['past_pred_ctxt']}predCtxt_{self._params['init_cond_ctxt']}initCtxt_{self._models[id]._which_out}_longSeq.png")           

            for plot in range(3):
                cat_mu = np.concatenate((val_output[2].error_low[:,plot,:,:,:],\
                                            val_output[2].error_high[:,plot,:,:,:]), axis=1)
                data_dict = {'cat_mu': cat_mu, "gr_tr": val_output[2].error[plot]}
                pickle.dump(data_dict, open(save_dir + '/error_val_data.pkl', 'wb'))        

        return 

    def val_hopfield(self, calib_data: Tuple[PICalibData], id: int, epoch:int):    
        """
        Validate the Modern Hopfield Network (MHN) after every few epochs.
        """
        #pbar = tqdm(self._val_loader, desc=f'Val Epoch {epoch}')
        self._models[id].eval()
        # size of final output pred
        wink_scores = []
        num_heads = self._params['num_heads']
        no_of_episodes = calib_data[3].Y.shape[0]
        alphas = len(self._models[id]._alphas)
        seq_length = calib_data[3].Y.shape[1]
        feat = 1 # for now
        val_out = ValOutput(y=np.zeros((no_of_episodes,seq_length,feat)),\
                            y_pred=np.zeros((no_of_episodes,seq_length,feat)),\
                            y_low=np.zeros((num_heads,no_of_episodes,alphas,seq_length,feat)),\
                            y_high=np.zeros((num_heads,no_of_episodes,alphas,seq_length,feat)))
        
        val_out_error = ValOutputError(error=np.zeros((no_of_episodes,seq_length,feat)),\
                                    error_low=np.zeros((num_heads,no_of_episodes,alphas,seq_length,feat)),\
                                    error_high=np.zeros((num_heads,no_of_episodes,alphas,seq_length,feat)))
                            

        with torch.no_grad():
            for batch_idx, val_data in enumerate(self._val_loader): 

                out = self._models[id].forward(val_data, train=False)
                wink_scores.append(out[0].detach().cpu().item())
                start_idx = batch_idx*val_data[0].shape[0] 
                end_idx = (batch_idx+1)*val_data[0].shape[0]
                if self._params['mhn_output'] == 'y':
                    # saving output pred with bounds and ground truth
                    val_out.y[start_idx:end_idx,:,:] = out[1][:,:,self._models[id]._which_out].unsqueeze(2).detach().cpu().numpy()
                    val_out.y_pred[start_idx:end_idx,:,:] = out[2][:,:,self._models[id]._which_out].unsqueeze(2).detach().cpu().numpy()
                    val_out.y_low[:,start_idx:end_idx,:,:,:] = out[3].detach().cpu().numpy()
                    val_out.y_high[:,start_idx:end_idx,:,:,:] = out[4].detach().cpu().numpy() 

                    val_out_error.error[start_idx:end_idx,:,:] = out[5].detach().cpu().numpy()
                    val_out_error.error_low[:,start_idx:end_idx,:,:,:] = out[6].detach().cpu().numpy()
                    val_out_error.error_high[:,start_idx:end_idx,:,:,:] = out[7].detach().cpu().numpy() 
                      
        if self._params['mhn_output'] == 'y':
            return np.array(wink_scores).mean(), val_out, val_out_error
    

    def eval_hopfield(self, calib_data: Tuple[PICalibData,...], id: int):
        """
        Evaluate the hopfield on val data with data in self._models[id]._mem_data
        """
        print(f"**************** Evaluation ****************")
        # encode the X from true train data for evaluation
        self._models[id]._mem_data = self.encode_train(calib_data, id)

        no_of_episodes = calib_data[3].Y.shape[0]
        num_heads = self._params['num_heads']
        alphas = len(self._models[id]._alphas)
        seq_length = calib_data[3].Y.shape[1]
        feat = 1 
        val_out = ValOutput(y=np.zeros((no_of_episodes,seq_length,feat)),\
                            y_pred=np.zeros((no_of_episodes,seq_length,feat)),\
                            y_low=np.zeros((num_heads,no_of_episodes,alphas,seq_length,feat)),\
                            y_high=np.zeros((num_heads,no_of_episodes,alphas,seq_length,feat)),)
        
        val_out_error = ValOutputError(error=np.zeros((no_of_episodes,seq_length,feat)),\
                            error_low=np.zeros((num_heads,no_of_episodes,alphas,seq_length,feat)),\
                            error_high=np.zeros((num_heads,no_of_episodes,alphas,seq_length,feat)))

        pbar = tqdm(self._val_loader, desc=f'Evaluation')
        self._models[id].eval()

        samples_processed = 0

        with torch.no_grad():
            for batch_idx, val_data in enumerate(pbar): 

                out = self._models[id].evaluate(val_data, batch_idx=batch_idx)
                start_idx = batch_idx*val_data[0].shape[0] 
                end_idx = (batch_idx+1)*val_data[0].shape[0]
                # saving output pred with bounds and ground truth
                val_out.y[start_idx:end_idx,:,:] = out[0][:,:,self._models[id]._which_out].unsqueeze(2).detach().cpu().numpy()
                val_out.y_pred[start_idx:end_idx,:,:] = out[1][:,:,self._models[id]._which_out].unsqueeze(2).detach().cpu().numpy()
                val_out.y_low[:,start_idx:end_idx,:,:,:] = out[2].detach().cpu().numpy()
                val_out.y_high[:,start_idx:end_idx,:,:,:] = out[3].detach().cpu().numpy()   

                val_out_error.error[start_idx:end_idx,:,:] = out[4].detach().cpu().numpy()
                val_out_error.error_low[:,start_idx:end_idx,:,:,:] = out[5].detach().cpu().numpy()
                val_out_error.error_high[:,start_idx:end_idx,:,:,:] = out[6].detach().cpu().numpy()  

                # want to have 32 examples for plotting 
                samples_processed += val_data[0].shape[0]
                if samples_processed >= 32:
                    break
        #print(val_out_error.error_low[:,1,:,150,:])        
        #print(val_out_error.error_high[:,1,:,150,:]) 
     
        all_cat_mu = []
        all_gr_tr = []   
        for plot in range(4):        
            cat_mu = np.concatenate((np.expand_dims(np.expand_dims(val_out.y_pred[plot],axis=0),axis=0).repeat(num_heads,axis=0),\
                                    val_out.y_low[:,plot,:,:,:],\
                                        val_out.y_high[:,plot,:,:,:]), axis=1) # [num_heads,no_of_alphas,seq_len,_which_out]
            gr_tr = val_out.y[plot] # [seq_len,_which_out]

            all_cat_mu.append(np.expand_dims(cat_mu, axis=0))
            all_gr_tr.append(np.expand_dims(gr_tr, axis=0))       

        all_cat_mu = np.concatenate(all_cat_mu, axis=0) # [examples,num_heads,no_of_alphas,seq_len,_which_out]
        all_gr_tr = np.concatenate(all_gr_tr, axis=0) # [examples,seq_len,_which_out]

        all_cat_mu_er = []
        all_gr_tr_er = []

        for plot in range(32):
            cat_mu = np.concatenate((val_out_error.error_low[:,plot,:,:,:],\
                                        val_out_error.error_high[:,plot,:,:,:]), axis=1)
            gr_tr = val_out_error.error[plot] # [seq_len,_which_out]
            
            all_cat_mu_er.append(np.expand_dims(cat_mu, axis=0))
            all_gr_tr_er.append(np.expand_dims(gr_tr, axis=0))  

        all_cat_mu_er = np.concatenate(all_cat_mu_er, axis=0) # [examples,num_heads,no_of_alphas,seq_len,_which_out]
        all_gr_tr_er = np.concatenate(all_gr_tr_er, axis=0) # [examples,seq_len,_which_out]             
            
        return (all_cat_mu, all_gr_tr, all_cat_mu_er, all_gr_tr_er)
    
    def encode_train(self, calib_data: Tuple[PICalibData], id: int) -> MemoryData:
        """
        Encode the training data for evaluation and fills the memory
        """
        mem_data = MemoryData(X_ctx_true_train_enc=torch.full(tuple(calib_data[0].X_ctx.shape), fill_value=float('inf')),
                            error_train=torch.full(tuple(calib_data[2].error.shape), fill_value=float('inf'))
                        )
        for i in range(calib_data[0].X_ctx.shape[0]):

            mem_data.X_ctx_true_train_enc[i] = \
                                self._models[id]._encoder(calib_data[0].X_ctx[i].to(device)).cpu()
            mem_data.error_train[i] = calib_data[2].error[i]
            
        return mem_data
    
    def save_data_clustering(self, calib_data: Tuple[PICalibData]) -> None:
        """Save normalized data for clustering in Jupyter notebook"""

        save_dir = f"./../../../my_folder/calib_data/{self._params['dataset_name']}_{self._params['past_ts_ctxt']}tsCtxt_{self._params['past_feat_ctxt']}featCtxt_{self._params['past_pred_ctxt']}predCtxt_{self._params['init_cond_ctxt']}initCtxt"
        data_dict = {"calib_train_true": calib_data[0],
                    "calib_val_true": calib_data[1],
                    "calib_train_sim": calib_data[2],
                    "calib_val_sim": calib_data[3]
                    }        
        pickle.dump(data_dict, open(save_dir + ".pkl", 'wb'))

        return 

    def all_train_hopfield(self, calib_data: Tuple[PICalibData]) -> None:
        """
        Trains separate MHN model for each neural network output
        """

        calib_data = self.mix_train_data(calib_data) 
        self.save_mix_data(calib_data)
        calib_data = self.mix_val_data(calib_data) 

        #calib_data = self.positional_encoding(calib_data)
        
        self._train_loader, self._val_loader = self._data_loaders(calib_data)
        # calculate the validation MSE before MHN training 
        self.calculate_val_mse(self._val_loader)
        self._save_model(calib_data, pre_train=True)

        for id, _ in self._models.items():

            self.train_hopfield(calib_data, id)

            self._save_model(calib_data, False, id)  

        print("ALL MHN MODELS ARE TRAINED AND SAVED")    

        return 
    
    def save_mix_data(self, calib_data: Tuple[PICalibData,...]) -> None:
        """Save mix training data for validation"""
        save_dir = f"./{self._params['dataset_name']}/mhn_model_weights/{self._params['past_ts_ctxt']}tsCtxt_{self._params['past_feat_ctxt']}featCtxt_{self._params['past_pred_ctxt']}predCtxt_{self._params['init_cond_ctxt']}initCtxt_{self._params['mhn_lr']}lr_{self._params['mhn_l2_reg']}l2_{self._params['mhn_epochs']}epch_{self._params['num_heads']}heads"

        check_or_make_folder(f"./{self._params['dataset_name']}")
        check_or_make_folder(f"./{self._params['dataset_name']}/mhn_model_weights")
        check_or_make_folder(save_dir)    
        
        print("MHN orig dataset is saved now!")
        # Saving MHN data 
        data_dict = {"calib_train_true": calib_data[0],
                    "calib_val_true": calib_data[1],
                    "calib_train_sim": calib_data[2],
                    "calib_val_sim": calib_data[3]
                    }
        pickle.dump(data_dict, open(save_dir + "/mhn_data_orig.pkl", 'wb'))

        return 

    def _save_model(self, calib_data: Tuple[PICalibData,...], pre_train: bool, id: Optional[int]=None) -> None:
        """
        To save model's state dict for later use
        """
        save_dir = f"./{self._params['dataset_name']}/mhn_model_weights/{self._params['past_ts_ctxt']}tsCtxt_{self._params['past_feat_ctxt']}featCtxt_{self._params['past_pred_ctxt']}predCtxt_{self._params['init_cond_ctxt']}initCtxt_{self._params['mhn_lr']}lr_{self._params['mhn_l2_reg']}l2_{self._params['mhn_epochs']}epch_{self._params['num_heads']}heads"
        models_state_dict = dict()

        if not pre_train:
            # Saving MHN weights
            for model_id, _ in self._models.items():
                if model_id <= id:
                    models_state_dict[model_id] = (self._current_best_losses[model_id],
                                                    self._current_best_wghts[model_id])

            torch.save(models_state_dict, save_dir + "/mhn_weights.pt")        

        else:
            check_or_make_folder(f"./{self._params['dataset_name']}")
            check_or_make_folder(f"./{self._params['dataset_name']}/mhn_model_weights")
            check_or_make_folder(save_dir)    
            
            print("MHN dataset is saved now!")
            # Saving MHN data 
            data_dict = {"calib_train_true": calib_data[0],
                        "calib_val_true": calib_data[1],
                        "calib_train_sim": calib_data[2],
                        "calib_val_sim": calib_data[3]
                        }
            pickle.dump(data_dict, open(save_dir + "/mhn_data.pkl", 'wb'))

        return 


    def all_eval_hopfield(self) -> None:
        """
        Evaluate all MHN using val data
        """
        calib_data = self._load_model()
        self._train_loader, self._val_loader = self._data_loaders(calib_data)
       
        all_cat_mu = []
        all_gr_tr = []

        all_cat_mu_er = []
        all_gr_tr_er = []

        for id, model in self._models.items():
            cat_gr = self.eval_hopfield(calib_data, id)
            all_cat_mu.append(cat_gr[0]) # type: ignore
            all_gr_tr.append(cat_gr[1]) # type: ignore

            all_cat_mu_er.append(cat_gr[2]) # type: ignore
            all_gr_tr_er.append(cat_gr[3]) # type: ignore

        #print(all_cat_mu_er[0][1,:,:,150,:])    

        all_cat_mu = np.concatenate(all_cat_mu, axis=-1) # [examples,num_heads,no_of_alphas,seq_len,_which_out]   
        all_gr_tr = np.concatenate(all_gr_tr, axis=-1) # [examples,seq_len,no_of_outputs]

        all_cat_mu_er = np.concatenate(all_cat_mu_er, axis=-1) # [examples,num_heads,no_of_alphas,seq_len,_which_out]   
        all_gr_tr_er = np.concatenate(all_gr_tr_er, axis=-1) # [examples,seq_len,no_of_outputs]

        # Try different conformal selection hyperparams
        save_dir = f"./{self._params['dataset_name']}/mhn_model_weights/{self._params['past_ts_ctxt']}tsCtxt_{self._params['past_feat_ctxt']}featCtxt_{self._params['past_pred_ctxt']}predCtxt_{self._params['init_cond_ctxt']}initCtxt_{self._params['mhn_lr']}lr_{self._params['mhn_l2_reg']}l2_{self._params['mhn_epochs']}epch_{self._params['num_heads']}heads"
        check_or_make_folder(save_dir + "/eval_figs/") 
        new_dir = f"{self._params['cp_sampling'][0]}_{self._params['cp_sampling'][1]}idx_{self._params['cp_aggregate']}_{self._params['cp_alphas']}alphas_{self._params['cp_replacement']}replace"
        check_or_make_folder(save_dir+"/eval_figs/"+new_dir) 
        fig_name = save_dir+"/eval_figs/"+new_dir

        for example in range(all_cat_mu.shape[0]):
            mu_pred = []
            for head in range(self._params['num_heads']):
                mu_pred.append(np.expand_dims(np.concatenate((np.expand_dims(np.mean(all_cat_mu[example][head][1:,:,:],\
                                                                    axis=0), axis=0), all_cat_mu[example][head]), axis=0), axis=0))
  
            mu_pred = np.concatenate(mu_pred, axis=0)    
            #for _ in range(2): # a hack to get the updated fonts on the second plot
            #    plot_part_mhn_many(mu_pred, all_gr_tr[example], all_gr_tr.shape[-1], fig_name=fig_name+f"/plot_{example}.png")

        data_dict = {'cat_mu': mu_pred, "gr_tr": all_gr_tr[example]}
        pickle.dump(data_dict, open(save_dir + '/eval_data_y.pkl', 'wb'))  

        # to plot different examples for first feature concatenate along last dim (remove it later)
        all_err = []
        all_gr = []

        for example in range(all_cat_mu_er.shape[0]):
            mu_pred = []
            for head in range(self._params['num_heads']):
                mu_pred.append(np.expand_dims(np.concatenate((np.expand_dims(np.mean(all_cat_mu_er[example][head],\
                                                    axis=0), axis=0), all_cat_mu_er[example][head]), axis=0), axis=0))
  
            mu_pred = np.concatenate(mu_pred, axis=0)   
            
            all_err.append(mu_pred)
            all_gr.append(all_gr_tr_er[example])  

        all_err = np.concatenate(all_err, axis=-1) # cat along batch dim
        all_gr = np.concatenate(all_gr, axis=-1)   

        #print(all_err.shape)
        #print(all_err[:,:,150,1])
        data_dict = {'cat_mu': all_err, "gr_tr": all_gr}
        pickle.dump(data_dict, open(save_dir +  "/eval_data_err.pkl", 'wb'))        

        return      


    def _load_model(self) -> Tuple[PICalibData]:
        """
        Loads MHN weights and data
        """
        save_dir = f"./{self._params['dataset_name']}/mhn_model_weights/{self._params['past_ts_ctxt']}tsCtxt_{self._params['past_feat_ctxt']}featCtxt_{self._params['past_pred_ctxt']}predCtxt_{self._params['init_cond_ctxt']}initCtxt_{self._params['mhn_lr']}lr_{self._params['mhn_l2_reg']}l2_{self._params['mhn_epochs']}epch_{self._params['num_heads']}heads"
        models_state_dict = torch.load(save_dir + "/mhn_weights.pt", map_location=device)

        # Load MHN weights
        for id, model in self._models.items():
            print(f"MHN Model {id+1} Loss is: {models_state_dict[id][0]}")
            self._models[id].load_state_dict(models_state_dict[id][1])

        # Load MHN data 
        # mhn_data_orig contains random in train and seq in val
        data_state_dict = pickle.load(open(save_dir + '/mhn_data_orig.pkl', 'rb'))
        #data_state_dict = pickle.load(open(save_dir + '/mhn_data_TrainVal_seq.pkl', 'rb'))
        calib_data = tuple((data_state_dict['calib_train_true'],
                           data_state_dict['calib_val_true'],
                           data_state_dict['calib_train_sim'],
                           data_state_dict['calib_val_sim']))
        
        return calib_data
    
    def _data_loaders(self, calib_data: Tuple[PICalibData,...]) -> Tuple[DataLoader, DataLoader]: 
        
        return DataLoader(TrainLoader(X_ctx_true=calib_data[2].X_ctx, # TRAIN 0 
                                    X_ctx_sim=calib_data[2].X_ctx,
                                    errors=calib_data[2].error), 
                                    batch_size=self._params['mhn_batch_size'], shuffle=True),\
                DataLoader(ValLoader(X_ctx_true=calib_data[3].X_ctx, # VAL 1 
                                    X_ctx_sim=calib_data[3].X_ctx, 
                                    errors=calib_data[3].error, 
                                    Y=calib_data[1].Y,
                                    Y_pred=calib_data[3].Y),
                                    batch_size=self._params['mhn_batch_size'], shuffle=False)
    
    def mix_train_data(self, calib_data: Tuple[PICalibData,...]) \
                                -> Tuple[PICalibData,...]:
        
        # For TRAIN Loader
        train_eps = calib_data[2].X_ctx.shape[0] 
        seq_len = calib_data[2].X_ctx.shape[1] 
        x_dim = calib_data[2].X_ctx.shape[2] 
        error_dim = calib_data[2].error.shape[2] 

        X_ctx_true = calib_data[0].X_ctx.reshape(train_eps*seq_len,x_dim) 
        X_ctx_sim = calib_data[2].X_ctx.reshape(train_eps*seq_len,x_dim) 
        error = calib_data[2].error.reshape(train_eps*seq_len,error_dim) 
        
        rand_idx = torch.randperm(train_eps*seq_len).reshape(-1)
        X_ctx_true = X_ctx_true[rand_idx]
        X_ctx_sim = X_ctx_sim[rand_idx]
        error = error[rand_idx]

        new_seq_len = 5000
        #new_no_of_eps = (train_eps*seq_len) // new_seq_len
        extra_points = (train_eps*seq_len) % new_seq_len

        X_ctx_true = X_ctx_true[:-extra_points].reshape(-1,new_seq_len,x_dim)
        X_ctx_sim = X_ctx_sim[:-extra_points].reshape(-1,new_seq_len,x_dim)
        error = error[:-extra_points].reshape(-1,new_seq_len,error_dim) 

        calib_data[0].X_ctx = X_ctx_true
        calib_data[2].X_ctx = X_ctx_sim
        calib_data[2].error = error

        return calib_data

    
    def mix_val_data(self, calib_data: Tuple[PICalibData,...]) \
                                -> Tuple[PICalibData,...]:
        """Mixes training data to build seq. of new lengths"""

        # For VAL data
        seq_len = calib_data[1].X_ctx.shape[1] 
        x_dim = calib_data[1].X_ctx.shape[2] 
        error_dim = calib_data[3].error.shape[2] 
        val_eps = calib_data[3].X_ctx.shape[0] 
        y_dim = calib_data[1].Y.shape[2]

        X_ctx_true = calib_data[1].X_ctx.reshape(val_eps*seq_len,x_dim) 
        X_ctx_sim = calib_data[3].X_ctx.reshape(val_eps*seq_len,x_dim) 
        error = calib_data[3].error.reshape(val_eps*seq_len,error_dim)        
        y = calib_data[1].Y.reshape(val_eps*seq_len,y_dim)
        y_pred = calib_data[3].Y.reshape(val_eps*seq_len,y_dim)

        rand_idx = torch.randperm(val_eps*seq_len).reshape(-1)
        X_ctx_true = X_ctx_true[rand_idx]
        X_ctx_sim = X_ctx_sim[rand_idx]
        error = error[rand_idx]        
        y = y[rand_idx]
        y_pred = y_pred[rand_idx]

        new_seq_len = 5000
        #new_no_of_eps = (train_eps*seq_len) // new_seq_len
        extra_points = (val_eps*seq_len) % new_seq_len

        X_ctx_true = X_ctx_true[:-extra_points].reshape(-1,new_seq_len,x_dim)
        X_ctx_sim = X_ctx_sim[:-extra_points].reshape(-1,new_seq_len,x_dim)
        error = error[:-extra_points].reshape(-1,new_seq_len,error_dim) 
        y = y[:-extra_points].reshape(-1,new_seq_len,y_dim)
        y_pred = y_pred[:-extra_points].reshape(-1,new_seq_len,y_dim)

        calib_data[1].X_ctx = X_ctx_true
        calib_data[3].X_ctx = X_ctx_sim
        calib_data[3].error = error        
        calib_data[1].Y = y
        calib_data[3].Y = y_pred    

        return calib_data
    
    def calculate_val_mse(self, val_loader: DataLoader):
        """Calculate VAL MSE for comparison with MSE later"""

        mses = []
        for batch_idx, val_data in enumerate(self._val_loader):
             
             outs = val_data[3].shape[2]
             mses.append(torch.mean((val_data[3].reshape(-1,outs)-val_data[4].reshape(-1,outs))**2, dim=0).unsqueeze(0))

        out_mses = torch.mean(torch.cat(mses, dim=0), dim=0)
        
        for i in range(out_mses.shape[0]):
            print(f"The output {i+1} VAL MSE is: {out_mses[i]}")
 