import torch
import numpy as np
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Scores():
    def __init__(self):
        self._alphas = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45]

    def get_scores(self, mean_pred_unnorm_y: torch.Tensor, var_pred_unnorm_y: torch.Tensor,\
                   unnorm_lower_mu: torch.Tensor, unnorm_upper_mu: torch.Tensor,\
                      all_true_unnorm_y: torch.Tensor):
        """
        :param mean_pred_unnorm_y: [batches,seq_len,y_dim]
        :param var_pred_unnorm_y: [batches,seq_len,y_dim]
        :param unnorm_lower_mu: [len(self._alphas),batch_size,seq_len,y_dim]
        :param unnorm_upper_mu: [len(self._alphas),batch_size,seq_len,y_dim]
        :param all_true_unnorm_y: [batch_size,seq_len,y_dim]
        """ 
        # CALIBRATION SCORES
        calib_scores = self.save_calib_scores(unnorm_lower_mu,unnorm_upper_mu,all_true_unnorm_y)
        wink_scores, pi_widths, mses, nlls = [], [], [], []
        # Calculate Winkler Scores, PI-width, MSE and NLL
        for i in range(all_true_unnorm_y.shape[-1]):
            all_true_unnorm_y_ = all_true_unnorm_y[:,:,i].unsqueeze(-1)
            mean_pred_unnorm_y_ = mean_pred_unnorm_y[:,:,i].unsqueeze(-1)
            var_pred_unnorm_y_ = var_pred_unnorm_y[:,:,i].unsqueeze(-1)
            unnorm_lower_mu_ = unnorm_lower_mu[:,:,:,i][:,:,:,np.newaxis]
            unnorm_upper_mu_ = unnorm_upper_mu[:,:,:,i][:,:,:,np.newaxis]
            # Winkler score and PI Width
            wink_score, pi_width = self.winkler_score(all_true_unnorm_y_, unnorm_lower_mu_, unnorm_upper_mu_)
            mse = self.mse(all_true_unnorm_y_, mean_pred_unnorm_y_)
            nll = self.nll(mean_pred_unnorm_y_, var_pred_unnorm_y_, all_true_unnorm_y_)
            wink_scores.append(wink_score.mean().detach().cpu().numpy()) # spits out 9 scores (1 for each alpha)
            pi_widths.append(pi_width.mean().detach().cpu().numpy()) # spits out 9 scores (1 for each alpha)
            mses.append(mse.detach().cpu().numpy())
            nlls.append(nll.detach().cpu().numpy())

        return (calib_scores.detach().cpu().numpy(), np.array(wink_scores),\
              np.array(pi_widths), np.array(mses), np.array(nlls))   

    def winkler_score(self, y: torch.Tensor, y_low: torch.Tensor, y_high: torch.Tensor, is_errors: bool = False):
        """
        Calculates winkler score given ground truth and upper/lower bounds
           
        :param y: [batch_size,seq_len,1] 
        :param y_low: [len(self._alphas),batch_size,seq_len,1]
        :param y_high: [len(self._alphas),batch_size,seq_len,1]
        :return          
        """
        out = y.shape[-1]
        y = y.reshape(-1,out)
        y_low = torch.Tensor(y_low.reshape(len(self._alphas),-1,out)).to(device)
        y_high = torch.Tensor(y_high.reshape(len(self._alphas),-1,out)).to(device)

        batch_size = y.shape[0]  

        score = torch.zeros((len(self._alphas))).to(device)
        pi_width = torch.zeros((len(self._alphas))).to(device)

        for alpha in range(len(self._alphas)): # for all quantiles
            width = (y_high[alpha]-y_low[alpha]).abs()
            undershoot = torch.lt(y, y_low[alpha]).long()
            overshoot = torch.gt(y, y_high[alpha]).long()
            alph = self._alphas[alpha]
            score[alpha] = (width + (undershoot * (y_low[alpha] - y) * 2/alph) +
                             (overshoot * (y - y_high[alpha]) * 2/alph)).sum()
            # taking mean first over batch dim to be consistent with the previous non-batch implementation
            score[alpha] = score[alpha] / batch_size
            pi_width[alpha] = width.sum() / batch_size

        return score, pi_width # Don't mean across alphas, KEEP EACH ALPHA SEPARATED   
    
    def nll(self, expected_y: torch.Tensor,\
             variance_y: torch.Tensor, true_y: torch.Tensor):
        """Calculate the negative log-likelihood of the sampled outputs """
        expected_y = expected_y.reshape(-1,expected_y.shape[-1])
        variance_y = variance_y.reshape(-1,variance_y.shape[-1])
        #variance_y = torch.clamp(variance_y.reshape(-1, variance_y.shape[-1]), min=1e-6)
        true_y = true_y.reshape(-1,true_y.shape[-1]) # stack all batches
        #print(expected_y[:6])
        #print(true_y[:6])
        #print(variance_y[:6])
        #sys.exit()

        nll = 0.5 * torch.log(2 * torch.pi * variance_y) + 0.5 * ((true_y - expected_y) ** 2 / variance_y)
   
        return nll.mean()    

    def mse(self, true_y: torch.Tensor, expected_y: torch.Tensor):
        """Calculate the mean squared error of the expected value of outputs"""
        expected_y = expected_y.reshape(-1,expected_y.shape[-1])
        true_y = true_y.reshape(-1,true_y.shape[-1]) # stack all batches

        return ((true_y - expected_y) ** 2).mean()  
    
    def calibration_score(self, lower_mu_quant: torch.Tensor, upper_mu_quant: torch.Tensor,\
                           true_error: torch.Tensor):
        """
        :param lower_mu_quant: [len(self._alphas),batch_size*seq_len]
        :param upper_mu_quant:[len(self._alphas),batch_size*seq_len]
        :param true_error:[batch_size,seq_len,1]
        Check if the expected frequency matches the observed frequency
        """
        true_error = true_error.reshape(-1,) # [batch_size*seq_len,1]
        total_points = true_error.shape[0]
        undershoots = torch.zeros(len(self._alphas)).to(device)
        overshoots = torch.zeros(len(self._alphas)).to(device)

        for alpha in range(len(self._alphas)): # for all quantiles
            
            undershoot = torch.lt(true_error, lower_mu_quant[alpha]).long()
            overshoot = torch.gt(true_error, upper_mu_quant[alpha]).long()

            undershoots[alpha] = torch.where(undershoot == 1)[0].shape[0]
            overshoots[alpha] = torch.where(overshoot == 1)[0].shape[0]
        
        bounded_errors = total_points - (undershoots+overshoots)
        bounded_errors = (bounded_errors / total_points).unsqueeze(-1) # [len(self._alphas),1]

        return bounded_errors     

    def save_calib_scores(self, lower_mu_quant: torch.Tensor, upper_mu_quant: torch.Tensor,\
                           true_error: torch.Tensor):
        """
        :param lower_mu_quant: [len(self._alphas),batch_size,seq_len,dp_outs]
        :param upper_mu_quant:[len(self._alphas),batch_size,seq_len,dp_outs]
        :param true_error:[batch_size,seq_len,dp_outs]
        Check if the expected frequency matches the observed frequency
        """  
        batch_sz = true_error.shape[0]
        seq_len = true_error.shape[1]
        outs = true_error.shape[-1]

        lower_mu_quant = torch.from_numpy(lower_mu_quant.reshape(-1,batch_sz*seq_len,outs)).to(device)
        upper_mu_quant = torch.from_numpy(upper_mu_quant.reshape(-1,batch_sz*seq_len,outs)).to(device)
        true_error = true_error.reshape(-1,outs).to(device)
        
        scores = []
        for out in range(outs):
            scores.append(self.calibration_score(lower_mu_quant[:,:,out],\
                                    upper_mu_quant[:,:,out], true_error[:,out]))

        scores = torch.cat(scores, axis=-1).unsqueeze(0)   

        return scores 