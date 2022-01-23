#=====================================================================================
# FileName: gnn.py
# Description: P2Loc model. 
#=====================================================================================

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

class AllEmbedding(nn.Module):
    def __init__(self,in_dim, embed_dim, device):
        super(AllEmbedding, self).__init__()
        self.feature_l = in_dim
        self.device = device
        self.embed_dim = embed_dim
#         self.m_feature = m_feature
        self.w_r = nn.Linear(self.feature_l, self.embed_dim)
    
    def get_weight(self,m_feature):
        self.weight = self.w_r(torch.tensor(m_feature.values, dtype=torch.float32)).to(self.device)
    
    def forward(self, nodes_v):
        if len(nodes_v.size())==0:
            to_feat = self.weight[nodes_v.to(self.device)].view(1,-1)
        else:
            to_feat = self.weight[nodes_v.to(self.device)]
        return to_feat

class CatEmbedding(nn.Module):
    def __init__(self, cat_ls, num_ls,category_origin_dimension, category_embedding_dimension,in_dim, embed_dim, device = "cpu"):
        super(CatEmbedding, self).__init__()
        self.cat_ls = cat_ls
        self.num_ls = num_ls
        self.embed_dim = embed_dim
        self.device = device
        
        embeddings = {}
        self.feature_l = in_dim
        
        for cat_val, cat_origin_dim, cat_embed_dim in list(zip(cat_ls, 
                                                               category_origin_dimension, 
                                                               category_embedding_dimension)):
            embedding = nn.Embedding(cat_origin_dim+1, cat_embed_dim).to(self.device)
            embeddings[cat_val] = embedding

#         for num_val in num_ls:
#             embeddings[num_val] = nn.Linear(1, 16).to(device)
        self.embeddings = embeddings
        self.w_r = nn.Linear(self.feature_l, self.embed_dim).to(self.device)
        
    def get_weight(self,m_feature):
        feat_ebd = torch.cat((m_feature[self.cat_ls].apply(lambda x:self.embeddings[x.name](
                    torch.tensor(x.values, dtype=torch.long).to(self.device))).tolist()
                   +[torch.tensor(m_feature[self.num_ls].values, dtype=torch.float32).to(self.device)]),1)
        self.weight = self.w_r(feat_ebd)
    
    def forward(self, nodes_v):
        if len(nodes_v.size())==0:
            to_feat = self.weight[nodes_v.to(self.device)].view(1,-1)
        else:
            to_feat = self.weight[nodes_v.to(self.device)]
        return to_feat
    
class Attention(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(Attention, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.att1 = nn.Linear(self.in_dims, self.out_dims)
        self.att2 = nn.Linear(self.out_dims, self.out_dims)
        self.att3 = nn.Linear(self.out_dims, 1)
        self.softmax = nn.Softmax(0)

    def forward(self, node1, u_rep, num_neighs):
        uv_reps = u_rep.repeat(num_neighs, 1)
        x = torch.cat((node1, uv_reps), 1)
        x = F.relu(self.att1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.att2(x))
        x = F.dropout(x, training=self.training)
        x = self.att3(x)
        att = F.softmax(x, dim=0)
        return att

class UV_Aggregator(nn.Module):
    def __init__(self, features, v2e, u2e, u_embed_dim, v_embed_dim, cuda="cpu", hasfeature = True):
        super(UV_Aggregator, self).__init__()

        self.features = features
        self.hasfeature = hasfeature
        self.v2e = v2e
        # self.r2e = r2e
        self.u2e = u2e
        self.device = cuda
        self.u_embed_dim = u_embed_dim
        self.v_embed_dim = v_embed_dim
#         self.w_r1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_r1 = nn.Linear(self.v_embed_dim+1, self.v_embed_dim)
        self.w_r2 = nn.Linear(self.v_embed_dim, self.v_embed_dim)
        self.att = Attention(self.v_embed_dim+self.u_embed_dim, self.v_embed_dim)

    def forward(self, nodes, history_uv, history_r):

        embed_matrix = torch.empty(len(history_uv), self.v_embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(history_uv)):
            history = history_uv[i]
            num_histroy_item = len(history)
            tmp_label = history_r[i]

            if self.hasfeature == True:
                feature_neigbhors = self.features(torch.LongTensor(list(history))).to(self.device)
                e_uv = torch.t(feature_neigbhors)
            else:
                e_uv = self.v2e(history.to(self.device))
            uv_rep = self.u2e(nodes[i].to(self.device))

#             e_r = self.r2e.weight[tmp_label]
            e_r = torch.Tensor(tmp_label).view(-1, 1).to(self.device)
            x = torch.cat((e_uv, e_r), 1)
            x = F.relu(self.w_r1(x))
            o_history = F.relu(self.w_r2(x))

            att_w = self.att(o_history, uv_rep, num_histroy_item)
            att_history = torch.mm(o_history.t(), att_w)
            att_history = att_history.t()

            embed_matrix[i] = att_history
        to_feats = embed_matrix
        return to_feats

class UV_Encoder(nn.Module):

    def __init__(self, features, u_embed_dim, v_embed_dim, history_uv_lists, history_r_lists, aggregator, cuda="cpu"):
        super(UV_Encoder, self).__init__()

        self.features = features
        self.history_uv_lists = history_uv_lists
        self.history_r_lists = history_r_lists
        self.aggregator = aggregator
        self.u_embed_dim = u_embed_dim
        self.v_embed_dim = v_embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(self.u_embed_dim+self.v_embed_dim, self.u_embed_dim)  #

    def forward(self, nodes):
        tmp_history_uv = []
        tmp_history_r = []
        for node in nodes:
            tmp_history_uv.append(self.history_uv_lists[int(node)])
            tmp_history_r.append(self.history_r_lists[int(node)])

        neigh_feats = self.aggregator.forward(nodes, tmp_history_uv, tmp_history_r)

        self_feats = self.features(nodes.to(self.device))
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined

class Social_Aggregator(nn.Module):

    def __init__(self, features, u2e, embed_dim, num_uu ,cuda="cpu", hasfeature = False):
        super(Social_Aggregator, self).__init__()

        self.features = features
        self.hasfeature = hasfeature
        self.device = cuda
        self.u2e = u2e
        self.embed_dim = embed_dim
        self.num_uu = num_uu
        self.w_r1 = nn.Linear(self.embed_dim+self.num_uu, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att = Attention(self.embed_dim*2, self.embed_dim)

    def forward(self, nodes, to_neighs, neighs_r):
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        for i in range(len(nodes)):
            tmp_adj = to_neighs[i]
            num_neighs = len(tmp_adj)
            u_rep = self.u2e(nodes[i].to(self.device))
            if num_neighs == 0:
                embed_matrix[i] = u_rep.view(1,-1)
                continue
            tmp_label = neighs_r[i]

            if self.hasfeature:
                feature_neigbhors = self.features(torch.LongTensor(list(tmp_adj)).to(self.device))
                e_u = torch.t(feature_neigbhors)
            else:
                e_u = self.u2e(torch.LongTensor(list(tmp_adj)).to(self.device))
            

            # e_r = self.r2e.weight[tmp_label]
            if self.num_uu == 1:
                e_r = torch.Tensor(tmp_label).view(-1, 1).to(self.device)
            else:
                e_r = torch.Tensor(tmp_label).to(self.device)
                
            x = torch.cat((e_u, e_r), 1)
            x = F.relu(self.w_r1(x))
            o_history = F.relu(self.w_r2(x))

            att_w = self.att(o_history, u_rep, num_neighs)
            att_history = torch.mm(o_history.t(), att_w).t()
            embed_matrix[i] = att_history
        to_feats = embed_matrix

        return to_feats

class Social_Encoder(nn.Module):

    def __init__(self, features, embed_dim, social_adj_lists,social_r_adj_lists, aggregator, base_model=None, cuda="cpu", hasfeature = False):
        super(Social_Encoder, self).__init__()

        self.features = features
        self.hasfeature = hasfeature
        self.social_adj_lists = social_adj_lists
        self.social_r_adj_lists = social_r_adj_lists
        self.aggregator = aggregator
        if base_model != None:
            self.base_model = base_model
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)  #

    def forward(self, nodes):

        to_neighs = []
        neighs_r = []
        for node in nodes:
            to_neighs.append(self.social_adj_lists[int(node)])
            neighs_r.append(self.social_r_adj_lists[int(node)])
        neigh_feats = self.aggregator.forward(nodes, to_neighs, neighs_r) 
        if self.hasfeature:
            self_feats = self.features(torch.LongTensor(nodes.cpu().numpy())).to(self.device)
            self_feats = self_feats.t()
        else:
            self_feats = self.features(nodes.to(self.device))
        
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined

class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.u_embed_dim = enc_u.embed_dim
        self.v_embed_dim = enc_v_history.embed_dim

        self.w_ur1 = nn.Linear(self.u_embed_dim, self.u_embed_dim)
        self.w_ur2 = nn.Linear(self.u_embed_dim, self.u_embed_dim)
        self.w_vr1 = nn.Linear(self.v_embed_dim, self.v_embed_dim)
        self.w_vr2 = nn.Linear(self.v_embed_dim, self.v_embed_dim)
        self.w_uv1 = nn.Linear(self.u_embed_dim+self.v_embed_dim, self.u_embed_dim)
        self.w_uv2 = nn.Linear(self.u_embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
#         self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.u_embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.v_embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.u_embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)


    def forward(self, nodes_u, nodes_v):
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v_history(nodes_v)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()
    
#     def loss(self, nodes_u, nodes_v, labels_list):
#         scores = self.forward(nodes_u, nodes_v)
#         return self.criterion(scores, labels_list)
    
class Combiner(nn.Module):
    def __init__(self, class_model, reg_model, epoch_number, device):
        super(Combiner, self).__init__()
        self.epoch_number = epoch_number
        self.class_model = class_model
        self.reg_model = reg_model
        self.device = device
        self.sm = nn.Sigmoid()
        self.criterion_class = nn.BCELoss()
        self.criterion_reg = nn.MSELoss()
        self.initilize_all_parameters()
    
    def initilize_all_parameters(self):
        self.alpha = 0.2
        if self.epoch_number in [90, 180]:
            self.div_epoch = 100 * (self.epoch_number // 100 + 1)
        else:
            self.div_epoch = self.epoch_number
    
    def reset_epoch(self, epoch):
        self.epoch = epoch
    
    def forward(self, nodes_u, nodes_v):
        class_result = self.sm(self.class_model.forward(nodes_u, nodes_v))
        reg_result = self.reg_model.forward(nodes_u, nodes_v)
        predicted = class_result.ge(0.5).float()
        reg_result[predicted==0] = 1
        return class_result, reg_result

    def l2_loss(self):
        l2_reg = torch.tensor(0.).to(self.device)
        for param in self.reg_model.parameters():
            l2_reg += torch.norm(param)
        return l2_reg
        
    def loss(self, nodes_u, nodes_v, label01_list, labels_list, l2_lambda):
        l = 1 - ((self.epoch - 1) / self.div_epoch) ** 2
        class_result, reg_result = self.forward(nodes_u, nodes_v)
        loss = l *10000* self.criterion_class(class_result, label01_list) + (1 - l) * self.criterion_reg(reg_result, labels_list) + l2_lambda*self.l2_loss()
        return loss
        