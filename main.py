#=====================================================================================
# FileName: main.py
# Description: Main file.
# input:
#   history_u_lists: couriers' related merchants list in historical data (courier-merchant graph), e.g. {'courier1':[merchant1, merchant2],'courier2':[...],...}
#   history_ur_lists: the travel time after the courier left the merchant(in history_u_lists) in historical data, e.g. {'courier1':[travel_time1, travel_time2],'courier2':[...],...}
#   history_v_lists: deprecated
#   history_vr_lists: deprecated
#   train_u: couriers list in training dataset, e.g. [courier1, courier2, ...]
#   train_v: merchants list in training dataset, e.g. [merchant1, merchant2, ...]
#   train_r: the travel time after the courier(in train_u) left the merchant(in train_v)
#   test_u: couriers list in test dataset, e.g. [courier1, courier2, ...]
#   test_v: merchants list in test dataset, e.g. [merchant1, merchant2, ...]
#   test_r: the travel time after the courier(in test_u) left the merchant(in test_v)
#   social_adj_lists: the courier-courier graph, e.g. {'courier1':[courier2, courier3],'courier2':[...],...}
#   social_r_adj_lists: the edge features in courier-courier graph (including travel time), e.g. {'courier1':[[feature1, feature2],[feature1, feature2]],'courier2':[...],...}
#   vv_adj_lists: the merchant-merchant graph, e.g. {'merchant1':[merchant2, merchant3],'merchant2':[...],...}
#   vvr_adj_lists: the edge features in merchant-merchant graph (including travel time), e.g. {'merchant1':[[feature1, feature2],[feature1, feature2]],'merchant2':[...],...}
#=====================================================================================

from train import *
import argparse

def make_args():
    parser = argparse.ArgumentParser(description='Interface for P2Loc framework')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N', help='input batch size for training')
    parser.add_argument('--merchant_embed_dim', type=int, default=32, metavar='N', help='embedding size')
    parser.add_argument('--courier_embed_dim', type=int, default=128, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--l2_lambda', type=float, default=0.000, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--dir_data', type=str, default='test_dataset', metavar='N', help='data name')
    parser.add_argument('--cat_flag', type=int, default=False, metavar='N', help='classification or regression')
    parser.add_argument('--pre_model', type=int, default=False, metavar='N', help='have pretrain model or not')
    parser.add_argument('--result_name', type=str, default='20211031', metavar='N', help='result name')
    parser.add_argument('--c_feature', type=int, default=True, metavar='N', help='add courier feature')
    parser.add_argument('--m_feature', type=int, default=True, metavar='N', help='add merchant feature')
    args = parser.parse_args()
    return args

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    args = make_args()

    path_data = '../data/' + args.dir_data + ".pickle"
    data_file = open(path_data, 'rb')

    (history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, 
            test_v, test_r, social_adj_lists,social_r_adj_lists,vv_adj_lists,vvr_adj_lists) = pickle.load(
            data_file)

    train_r01 = np.where(np.array(train_r)==1,0,1).tolist()
    test_r01 = np.where(np.array(test_r)==1,0,1).tolist()
    ss = None

    num_users = max(social_adj_lists.keys())+1
    num_items = max(vv_adj_lists.keys())+1

    # len(social_r_adj_lists[0][0])
    num_uu = 0 
    for val in social_r_adj_lists.values():
        if len(val) != 0:
            num_uu = len(val[0])
            break
    # num_uu = 11 # len(social_r_adj_lists[0][0])
    num_vv = len(vvr_adj_lists[0][0])

    u2e = nn.Embedding(num_users, args.courier_embed_dim).to(device)
    
    v2e = nn.Embedding(num_items, args.merchant_embed_dim).to(device)

    # agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    # enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)

    agg_v_history = Social_Aggregator(None, v2e, args.merchant_embed_dim, num_vv, cuda=device)
    enc_v_history2 = Social_Encoder(v2e, args.merchant_embed_dim, vv_adj_lists,vvr_adj_lists, agg_v_history, base_model=None, cuda=device)

    # agg_v_history2 = Social_Aggregator(lambda nodes: enc_v_history(nodes).t(), v2e, embed_dim, num_vv, cuda=device, hasfeature=True)
    # enc_v_history2 = Social_Encoder(lambda nodes: enc_v_history(nodes).t(), embed_dim, vv_adj_lists,vvr_adj_lists, agg_v_history,
    #                     base_model=enc_v_history, cuda=device, hasfeature=True)

    agg_u_history = UV_Aggregator(lambda nodes: enc_v_history2(nodes).t(), v2e, u2e, args.courier_embed_dim, args.merchant_embed_dim, cuda=device)
    enc_u_history = UV_Encoder(u2e, args.courier_embed_dim, args.merchant_embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device)
    # neighobrs
    agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e, args.courier_embed_dim, num_uu, cuda=device, hasfeature=True)
    enc_u = Social_Encoder(lambda nodes: enc_u_history(nodes).t(), args.courier_embed_dim, social_adj_lists,social_r_adj_lists, agg_u_social,
                        base_model=enc_u_history, cuda=device, hasfeature=True)

    # get train and test data
    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                            torch.FloatTensor(train_r01),torch.FloatTensor(train_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                            torch.FloatTensor(test_r01), torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)

    # model
    cls_graphrec = GraphRec(enc_u, enc_v_history2).to(device)
    reg_graphrec = GraphRec(enc_u, enc_v_history2).to(device)
    combiner = Combiner(cls_graphrec, reg_graphrec, args.epochs, device).to(device)

    optimizer = torch.optim.RMSprop(combiner.parameters(), lr=args.lr, alpha=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    best_medae = 9999.0
    best_auc = -1
    endure_count = 0

    result_path = args.dir_data+args.result_name
    for epoch in range(1, args.epochs + 1):
        combiner.reset_epoch(epoch)
        train(combiner, device, train_loader, optimizer, epoch, best_rmse, best_mae, best_medae,best_auc,args)
        expected_rmse, mae, medae, auc, u_ls, v_ls, target, preds = test(combiner, device, test_loader, ss)

        if best_mae > mae:
        # if best_medae > medae:
            best_rmse = expected_rmse
            best_mae = mae
            best_medae = medae
            best_auc = auc
            endure_count = 0
            result = pd.DataFrame({'u':u_ls, 'v':v_ls, 'target':target, 'preds':preds})
            result.to_csv('../result/'+result_path+'.csv',index = False)
        else:
            endure_count += 1
        print(datetime.datetime.now(), "| rmse: %.4f, mae:%.4f, medae:%.4f, auc:%.4f,  " % (expected_rmse, mae, medae, best_auc))

        if endure_count > 5:
            break

    model_path = result_path
    torch.save({
                'epoch': epoch,
                'model_state_dict': combiner.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, '../Model/'+model_path+'_model.pkl')

if __name__ == '__main__':
    main()