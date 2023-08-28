from transformers import RobertaModel, RobertaConfig, RobertaForSequenceClassification
from transformers import BertModel, BertConfig, BertForSequenceClassification
from transformers import XLNetModel, XLNetConfig, XLNetForSequenceClassification
import torch.nn.functional as F
import torch.nn as nn
import copy
import torch
from supConloss import SupConLoss


def make_model(args,device):
  if args.model == "roberta":
    config = RobertaConfig.from_pretrained("roberta-base")
    config.num_labels = 3
    if args.task == "sst":
      config.num_labels = 2
    if args.task == "ag_news":
      config.num_labels = 4
    if args.task == "yahoo":
      config.num_labels = 10
    if args.task == "mytask":
      config.num_labels = 2
    if args.task == "csn" or args.task == "csn-ambig" or args.task == "reactions" or args.task == "explorations" or args.task == "interpretations":
          num_labels = 3
    pretrained_model = RobertaForSequenceClassification.from_pretrained("roberta-base",config=config)
    return scl_model_Roberta(config,device,pretrained_model,with_semi=args.with_mix,with_sum=args.with_summary)

  if args.model == "bert":
    config = BertConfig.from_pretrained("bert-base-uncased")
    config.num_labels = 3
    if args.task == "imdb":
      config.num_labels = 2
    if args.task == "ag_news":
      config.num_labels = 4
    if args.task == "yahoo":
      config.num_labels = 10
    if args.task == "mytask":
      config.num_labels = 2
    if args.task == "csn" or args.task == "csn-ambig" or args.task == "reactions" or args.task == "explorations" or args.task == "interpretations":
          num_labels = 3
    pretrained_model = BertForSequenceClassification.from_pretrained("bert-base-uncased",config=config)
    #return scl_model_Bert(config,device,pretrained_model,with_semi=args.with_mix,with_sum=args.with_summary)
    return scl_model_Bert(config,device,pretrained_model,with_semi=args.with_mix,with_sum=args.with_summary)

  if args.model == "xlnet":
    config = XLNetConfig.from_pretrained("xlnet-base-cased")
    config.num_labels = 3
    if args.task == "imdb":
      config.num_labels = 2
    if args.task == "ag_news":
      config.num_labels = 4
    if args.task == "yahoo":
      config.num_labels = 10
    if args.task == "mytask":
      config.num_labels = 2
      
    if args.task == "csn" or args.task == "csn-ambig" or args.task == "reactions" or args.task == "explorations" or args.task == "interpretations":
          num_labels = 3
    pretrained_model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased",config=config)
    return scl_model_Xlnet(config,device,pretrained_model,with_semi=args.with_mix,with_sum=args.with_summary)


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class scl_model_Roberta(nn.Module):
    def __init__(self,config,device,pretrained_model,with_semi=True,with_sum=True):
        super().__init__()
        self.cls_x = ClassificationHead(config)
        self.cls_s = ClassificationHead(config)
        self.mlp_x = nn.Sequential(nn.Linear(config.hidden_size,config.hidden_size),nn.ReLU(),nn.Linear(config.hidden_size,256))
        self.mlp_s = nn.Sequential(nn.Linear(config.hidden_size,config.hidden_size),nn.ReLU(),nn.Linear(config.hidden_size,256))

        self.f = RobertaModel(config, add_pooling_layer=False)
        self.scl_criterion = SupConLoss(temperature=0.3,base_temperature = 0.3)
        self.ce_criterion = nn.CrossEntropyLoss()
        # self.f = copy.deepcopy(pretrained_enc)

        # self.f = RobertaModel(config)
        self.device = device
        self.init_weights(pretrained_model)
        self.with_semi = with_semi
        self.with_sum = with_sum
    def init_weights(self,pretrained_model):
        self.cls_x = copy.deepcopy(pretrained_model.classifier)
        self.cls_s = copy.deepcopy(pretrained_model.classifier)
        self.f = copy.deepcopy(pretrained_model.roberta)
        for p in self.mlp_x.parameters():
            # if p.dim() > 2:
                # torch.nn.init.xavier_normal_(p)
            torch.nn.init.normal_(p)

        for p in self.mlp_s.parameters():
            # if p.dim() > 2:
                # torch.nn.init.xavier_normal_(p)
            torch.nn.init.normal_(p)

    def predict(self,x):
        f_x = self.f(x)[0]

        score = self.cls_x(f_x)
        return score
    def forward(self,batch):
        x,s_mix,y_a,y_b = [item.to(self.device) for item in batch]
        f_x = self.f(x)[0]
        f_s = self.f(s_mix)[0]

        p_x = self.cls_x(f_x)
        # p_s = self.cls_s(f_s)
        p_s = self.cls_x(f_s)
        
        # print(p_x.shape)
        # print(y_a.shape)

        ce_loss_x = self.ce_criterion(p_x,y_a)
        if self.with_semi:
            ce_loss_s = (self.ce_criterion(p_s,y_a) + self.ce_criterion(p_s,y_b)) / 2
        else:
            ce_loss_s = self.ce_criterion(p_s,y_a)

        z_x = self.mlp_x(f_x[:,0,:]).unsqueeze(1)
        # z_x = f_x[:,0,:].unsqueeze(1)
        # z_s = self.mlp_s(f_s[:,0,:]).unsqueeze(1)
        z_s = self.mlp_x(f_s[:,0,:]).unsqueeze(1)

        # z_s = f_s[:,0,:].unsqueeze(1)
        if self.with_sum:
          z = torch.cat([z_x,z_s],dim=1)
        else:
          z = z_x

        if self.with_semi:
            scl_loss = (self.scl_criterion(z,labels = y_a) + self.scl_criterion(z,labels = y_b)) / 2
        else:
            scl_loss = self.scl_criterion(z,labels = y_a)

        

        return ce_loss_x, ce_loss_s, scl_loss
        
        
        
    def forward_corssentropy_alone(self,batch):
       
        x,s_mix,y_a,y_b = [item.to(self.device) for item in batch]
        f_x = self.f(x)[0]

        p_x = self.cls_x(f_x)
        
        # print(p_x.shape)
        # print(y_a.shape)

        ce_loss_x = self.ce_criterion(p_x,y_a)
    

        z_x = self.mlp_x(f_x[:,0,:]).unsqueeze(1)
        # z_x = f_x[:,0,:].unsqueeze(1)

        # z_s = f_s[:,0,:].unsqueeze(1)
        if self.with_sum:
          z = torch.cat([z_x,z_s],dim=1)
        else:
          z = z_x

        

        

        return ce_loss_x
        
        
        
        
    def forward_feature_mix_ce(self,batch):
       
        x,x_perm,s,s_perm,y_a,y_b = [item.to(self.device) for item in batch]
        f_x = 0.5 * self.f(x)[0] + 0.5 * self.f(x_perm)[0]
        f_s = 0.5 * self.f(s)[0] + 0.5 * self.f(s_perm)[0]

        p_x = self.cls_x(f_x)
        # p_s = self.cls_s(f_s)
        p_s = self.cls_x(f_s)
        
        # print(p_x.shape)
        # print(y_a.shape)
   
        
        ce_loss_s = (self.ce_criterion(p_s,y_a) + self.ce_criterion(p_s,y_b)) / 2
        ce_loss_x = (self.ce_criterion(p_x,y_a) + self.ce_criterion(p_x,y_b)) / 2

        

        return ce_loss_x, ce_loss_s


        

        return ce_loss_x, ce_loss_s
        
        
    def forward_feature_mix(self,batch):

        x,x_perm,s,s_perm,y_a,y_b = [item.to(self.device) for item in batch]
        f_x = 0.5 * self.f(x)[0] + 0.5 * self.f(x_perm)[0]
        f_s = 0.5 * self.f(s)[0] + 0.5 * self.f(s_perm)[0]

        p_x = self.cls_x(f_x)
        # p_s = self.cls_s(f_s)
        p_s = self.cls_x(f_s)
        
        # print(p_x.shape)
        # print(y_a.shape)
   
        
        ce_loss_s = (self.ce_criterion(p_s,y_a) + self.ce_criterion(p_s,y_b)) / 2
        ce_loss_x = (self.ce_criterion(p_x,y_a) + self.ce_criterion(p_x,y_b)) / 2

        z_x = self.mlp_x(f_x[:,0,:]).unsqueeze(1)
        # z_x = f_x[:,0,:].unsqueeze(1)
        # z_s = self.mlp_s(f_s[:,0,:]).unsqueeze(1)
        z_s = self.mlp_x(f_s[:,0,:]).unsqueeze(1)

        # z_s = f_s[:,0,:].unsqueeze(1)
        z = torch.cat([z_x,z_s],dim=1)

        scl_loss = (self.scl_criterion(z,labels = y_a) + self.scl_criterion(z,labels = y_b)) / 2

        

        return ce_loss_x, ce_loss_s, scl_loss


'''


The function `forward_feature_mix` represents a forward pass in a neural network model. It takes a `batch` of data as input and performs several computations to generate different loss values. Here's a breakdown of what each line does:

1. `x, x_perm, s, s_perm, y_a, y_b = [item.to(self.device) for item in batch]`: Assigns the elements of the input `batch` to variables `x`, `x_perm`, `s`, `s_perm`, `y_a`, and `y_b`. This line assumes that the `batch` is a list or tuple containing tensors and moves these tensors to the device specified by `self.device`.

2. `f_x = 0.5 * self.f(x)[0] + 0.5 * self.f(x_perm)[0]`: Passes the tensors `x` and `x_perm` through a model `self.f` and obtains their outputs. The outputs are then combined using a weighted sum. This line computes the feature representations `f_x` by combining the features from the original input `x` and the shuffled input `x_perm`.

3. `f_s = 0.5 * self.f(s)[0] + 0.5 * self.f(s_perm)[0]`: Passes the tensors `s` and `s_perm` through the same model `self.f` and obtains their outputs. The outputs are combined using a weighted sum. This line computes the feature representations `f_s` by combining the features from the original input `s` and the shuffled input `s_perm`.

4. `p_x = self.cls_x(f_x)`: Passes the feature representations `f_x` through another model `self.cls_x` to obtain predictions `p_x` for the original input.

5. `p_s = self.cls_x(f_s)`: Passes the feature representations `f_s` through the same model `self.cls_x` to obtain predictions `p_s` for the shuffled input.

6. `ce_loss_s = (self.ce_criterion(p_s,y_a) + self.ce_criterion(p_s,y_b)) / 2`: Computes the cross-entropy loss `ce_loss_s` by comparing the predictions `p_s` with the labels `y_a` and `y_b`. The loss is averaged between the two sets of labels.

7. `ce_loss_x = (self.ce_criterion(p_x,y_a) + self.ce_criterion(p_x,y_b)) / 2`: Computes the cross-entropy loss `ce_loss_x` by comparing the predictions `p_x` with the labels `y_a` and `y_b`. The loss is averaged between the two sets of labels.

8. `z_x = self.mlp_x(f_x[:,0,:]).unsqueeze(1)`: Extracts a specific dimension from the feature representations `f_x`, passes it through a multi-layer perceptron (MLP) model `self.mlp_x`, and then unsqueezes it to add an extra dimension. The result is assigned to `z_x`.

9. `z_s = self.mlp_x(f_s[:,0,:]).unsqueeze(1)`: Extracts a specific dimension from the feature representations `f_s`, passes it through the same MLP model `self.mlp_x`, and unsqueezes it to add an extra dimension. The result is assigned to `z_s`.

10. `z = torch.cat([z_x, z_s], dim=1)`: Concatenates the tensors `z_x` and `z_s` along the specified dimension (dimension 1) to obtain the final tensor `z`.

11. `scl_loss = (self.scl_criterion(z

, labels=y_a) + self.scl_criterion(z, labels=y_b)) / 2`: Computes the similarity contrastive loss `scl_loss` by comparing the tensor `z` with the labels `y_a` and `y_b`. The loss is averaged between the two sets of labels.

12. Finally, the function returns `ce_loss_x`, `ce_loss_s`, and `scl_loss`, representing the computed cross-entropy losses for the original and shuffled inputs, and the similarity contrastive loss, respectively.

'''


####################### new change #############################

    def forward_feature_mix_new(self,batch):

        x,x_perm,s,s_perm,y_a,y_b = [item.to(self.device) for item in batch]
        f_x = 0.5 * self.f(x)[0] + 0.5 * self.f(x_perm)[0]
        f_s = 0.5 * self.f(s)[0] + 0.5 * self.f(s_perm)[0]

        p_x = self.cls_x(f_x)
        p_s = self.cls_x(f_s)
        
        ce_loss_s = (self.ce_criterion(p_s,y_a) + self.ce_criterion(p_s,y_b)) / 2
        ce_loss_x = (self.ce_criterion(p_x,y_a) + self.ce_criterion(p_x,y_b)) / 2

        z_x = self.mlp_x(f_x[:,0,:]).unsqueeze(1)
        z_s = self.mlp_x(f_s[:,0,:]).unsqueeze(1)
        z = torch.cat([z_x,z_s],dim=1)

        # Initialize scl_loss as an empty tensor
        scl_loss = torch.empty(0, device=self.device)

        # Iterate over pairs of elements in y_a and y_b
        for i in range(y_a.size(0)):
            #print('8888888888888888', y_a[i], y_b[i])
            if y_a[i] == y_b[i]:
                # Append current loss to the scl_loss tensor
                curr_loss = self.scl_criterion(z[i].unsqueeze(0), labels=y_a[i].unsqueeze(0))  # Added unsqueeze(0)
                scl_loss = torch.cat((scl_loss, curr_loss.view(1)), dim=0)

        # Handle the case when scl_loss is still empty (no equal elements found)
        if scl_loss.nelement() == 0:
            scl_loss = torch.tensor(0, dtype=torch.float, device=self.device)

        return ce_loss_x, ce_loss_s, scl_loss.mean()
#########################################################################

class scl_model_Xlnet(nn.Module):
    def __init__(self,config,device,pretrained_model,with_semi=True,with_sum=True):
        super().__init__()
        self.cls_x = nn.Linear(config.d_model, config.num_labels)
        self.cls_s = nn.Linear(config.d_model, config.num_labels)
        self.mlp_x = nn.Sequential(nn.Linear(config.hidden_size,config.hidden_size),nn.ReLU(),nn.Linear(config.hidden_size,256))
        self.mlp_s = nn.Sequential(nn.Linear(config.hidden_size,config.hidden_size),nn.ReLU(),nn.Linear(config.hidden_size,256))

        self.f = XLNetModel(config)
        self.scl_criterion = SupConLoss(temperature=0.3,base_temperature = 0.3)
        self.ce_criterion = nn.CrossEntropyLoss()
        # self.f = copy.deepcopy(pretrained_enc)

        # self.f = RobertaModel(config)
        self.device = device
        self.init_weights(pretrained_model)
        self.with_semi = with_semi
        self.with_sum = with_sum
    def init_weights(self,pretrained_model):
        self.cls_x = copy.deepcopy(pretrained_model.logits_proj)
        self.cls_s = copy.deepcopy(pretrained_model.logits_proj)
        self.f = copy.deepcopy(pretrained_model.transformer)
          
        for p in self.mlp_x.parameters():
            torch.nn.init.normal_(p)

        for p in self.mlp_s.parameters():
            torch.nn.init.normal_(p)

    def predict(self,x):
        f_x = self.f(x)[0]
        score = self.cls_x(f_x[:,0,:])
        return score
    def forward(self,batch):
        x,s_mix,y_a,y_b = [item.to(self.device) for item in batch]
        f_x = self.f(x)[0]
        f_s = self.f(s_mix)[0]

        p_x = self.cls_x(f_x[:,0,:])
        # p_s = self.cls_s(f_s)
        p_s = self.cls_x(f_s[:,0,:])
        
        # print(p_x.shape)
        # print(y_a.shape)
        ce_loss_x = self.ce_criterion(p_x,y_a)
        if self.with_semi:
            ce_loss_s = (self.ce_criterion(p_s,y_a) + self.ce_criterion(p_s,y_b)) / 2
        else:
            ce_loss_s = self.ce_criterion(p_s,y_a)

        z_x = self.mlp_x(f_x[:,0,:]).unsqueeze(1)
        # z_x = f_x[:,0,:].unsqueeze(1)
        # z_s = self.mlp_s(f_s[:,0,:]).unsqueeze(1)
        z_s = self.mlp_x(f_s[:,0,:]).unsqueeze(1)

        # z_s = f_s[:,0,:].unsqueeze(1)
        if self.with_sum:
          z = torch.cat([z_x,z_s],dim=1)
        else:
          z = z_x

        if self.with_semi:
            scl_loss = (self.scl_criterion(z,labels = y_a) + self.scl_criterion(z,labels = y_b)) / 2
        else:
            scl_loss = self.scl_criterion(z,labels = y_a)

        

        return ce_loss_x, ce_loss_s, scl_loss

    
    def forward_feature_mix(self,batch):

        x,x_perm,s,s_perm,y_a,y_b = [item.to(self.device) for item in batch]
        f_x = 0.5 * self.f(x)[0] + 0.5 * self.f(x_perm)[0]
        f_s = 0.5 * self.f(s)[0] + 0.5 * self.f(s_perm)[0]

        p_x = self.cls_x(f_x)
        # p_s = self.cls_s(f_s)
        p_s = self.cls_x(f_s)
        
        # print(p_x.shape)
        # print(y_a.shape)

        
        ce_loss_s = (self.ce_criterion(p_s,y_a) + self.ce_criterion(p_s,y_b)) / 2
        ce_loss_x = (self.ce_criterion(p_x,y_a) + self.ce_criterion(p_x,y_b)) / 2

        z_x = self.mlp_x(f_x[:,0,:]).unsqueeze(1)
        # z_x = f_x[:,0,:].unsqueeze(1)
        # z_s = self.mlp_s(f_s[:,0,:]).unsqueeze(1)
        z_s = self.mlp_x(f_s[:,0,:]).unsqueeze(1)

        # z_s = f_s[:,0,:].unsqueeze(1)
        z = torch.cat([z_x,z_s],dim=1)

        scl_loss = (self.scl_criterion(z,labels = y_a) + self.scl_criterion(z,labels = y_b)) / 2

        

        return ce_loss_x, ce_loss_s, scl_loss


class scl_model_Bert(nn.Module):
    def __init__(self,config,device,pretrained_model,with_semi=True,with_sum=True):
        super().__init__()
        self.cls_x = nn.Linear(config.hidden_size, config.num_labels)
        self.cls_s = nn.Linear(config.hidden_size, config.num_labels)

        self.dropout_x = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_s = nn.Dropout(config.hidden_dropout_prob)

        self.mlp_x = nn.Sequential(nn.Linear(config.hidden_size,config.hidden_size),nn.ReLU(),nn.Linear(config.hidden_size,256))
        self.mlp_s = nn.Sequential(nn.Linear(config.hidden_size,config.hidden_size),nn.ReLU(),nn.Linear(config.hidden_size,256))

        self.f = BertModel(config)
        self.scl_criterion = SupConLoss(temperature=0.3,base_temperature = 0.3)
        self.ce_criterion = nn.CrossEntropyLoss()
        # self.f = copy.deepcopy(pretrained_enc)

        # self.f = RobertaModel(config)
        self.device = device
        self.init_weights(pretrained_model)
        self.with_semi = with_semi
        self.with_sum = with_sum
    def init_weights(self,pretrained_model):
        self.cls_x = copy.deepcopy(pretrained_model.classifier)
        self.cls_s = copy.deepcopy(pretrained_model.classifier)
        self.f = copy.deepcopy(pretrained_model.bert)
          
        for p in self.mlp_x.parameters():
            torch.nn.init.normal_(p)

        for p in self.mlp_s.parameters():
            torch.nn.init.normal_(p)

    def predict(self,x):
        f_x = self.f(x)[0]
        score = self.cls_x(f_x[:,0,:])
        return score
    def forward(self,batch):
        x,s_mix,y_a,y_b = [item.to(self.device) for item in batch]
        f_x = self.f(x)[0]
        f_s = self.f(s_mix)[0]

        p_x = self.cls_x(f_x[:,0,:])
        # p_s = self.cls_s(f_s)
        p_s = self.cls_x(f_s[:,0,:])
        
        # print(p_x.shape)
        # print(y_a.shape)

        ce_loss_x = self.ce_criterion(p_x,y_a)
        if self.with_semi:
            ce_loss_s = (self.ce_criterion(p_s,y_a) + self.ce_criterion(p_s,y_b)) / 2
        else:
            ce_loss_s = self.ce_criterion(p_s,y_a)

        z_x = self.mlp_x(f_x[:,0,:]).unsqueeze(1)
        # z_x = f_x[:,0,:].unsqueeze(1)
        # z_s = self.mlp_s(f_s[:,0,:]).unsqueeze(1)
        z_s = self.mlp_x(f_s[:,0,:]).unsqueeze(1)

        # z_s = f_s[:,0,:].unsqueeze(1)
        if self.with_sum:
          z = torch.cat([z_x,z_s],dim=1)
        else:
          z = z_x

        if self.with_semi:
            scl_loss = (self.scl_criterion(z,labels = y_a) + self.scl_criterion(z,labels = y_b)) / 2
        else:
            scl_loss = self.scl_criterion(z,labels = y_a)

        

        return ce_loss_x, ce_loss_s, scl_loss

        

        return ce_loss_x, ce_loss_s, scl_loss




    def forward_corssentropy_alone(self,batch):
       
        x,s_mix,y_a,y_b = [item.to(self.device) for item in batch]
        f_x = self.f(x)[0]

        p_x = self.cls_x(f_x)
        p_x = p_x[:,0,:] # mahshid added this to remove seq length 200 from dim

        # print(p_x.shape)
        # print(y_a.shape)

        ce_loss_x = self.ce_criterion(p_x,y_a)
    

        z_x = self.mlp_x(f_x[:,0,:]).unsqueeze(1)
        # z_x = f_x[:,0,:].unsqueeze(1)

        # z_s = f_s[:,0,:].unsqueeze(1)
        if self.with_sum:
          z = torch.cat([z_x,z_s],dim=1)
        else:
          z = z_x

        

        

        return ce_loss_x







    def forward_feature_mix(self,batch):

        x,x_perm,s,s_perm,y_a,y_b = [item.to(self.device) for item in batch]
        f_x = 0.5 * self.f(x)[0] + 0.5 * self.f(x_perm)[0]
        f_s = 0.5 * self.f(s)[0] + 0.5 * self.f(s_perm)[0]

        p_x = self.cls_x(f_x)
        # p_s = self.cls_s(f_s)
        p_s = self.cls_x(f_s)
        
        p_x = p_x[:,0,:] # mahshid added this to remove seq length 200 from dim
        p_s = p_s[:,0,:] # mahshid added this to remove seq length 200 from dim

        ce_loss_s = (self.ce_criterion(p_s,y_a) + self.ce_criterion(p_s,y_b)) / 2
        ce_loss_x = (self.ce_criterion(p_x,y_a) + self.ce_criterion(p_x,y_b)) / 2

        z_x = self.mlp_x(f_x[:,0,:]).unsqueeze(1)
        # z_x = f_x[:,0,:].unsqueeze(1)
        # z_s = self.mlp_s(f_s[:,0,:]).unsqueeze(1)
        z_s = self.mlp_x(f_s[:,0,:]).unsqueeze(1)

        # z_s = f_s[:,0,:].unsqueeze(1)
        z = torch.cat([z_x,z_s],dim=1)

        scl_loss = (self.scl_criterion(z,labels = y_a) + self.scl_criterion(z,labels = y_b)) / 2

        

        return ce_loss_x, ce_loss_s, scl_loss



