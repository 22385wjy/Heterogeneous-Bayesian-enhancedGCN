import os
import torch
import numpy as np

from opt import OptInit
from hbGCN import DBwGcn
from assess.metrics import accuracy, auc, prf
from dataSet import dataLoader


if __name__ == '__main__':
    opt = OptInit().initialize()

    ### Load data
    dl = dataLoader()
    raw_features, y, nonimg = dl.load_data()
    #print(raw_features.shape, y.shape, nonimg.shape)
    n_folds = opt.fd
    cv_splits = dl.data_split(n_folds)
    corrects = np.zeros(n_folds, dtype=np.int32)
    accs = np.zeros(n_folds, dtype=np.float32) 
    aucs = np.zeros(n_folds, dtype=np.float32)
    prfs = np.zeros([n_folds,3], dtype=np.float32)
    
    save_train_val = open(opt.saveLossAcc, 'a')
    save_train_val.write(
        'Epoch' + ' ' + 'Train_Loss' + ' ' + 'Train_Accuracy' + ' ' + 'Valid_Loss' + ' ' + 'Valid__Accuracy' + ' ' + '\n')
    save_evalustion_folds = open(opt.saveFoldOut, 'a')
    save_evalustion_folds.write(
        'stage'+' '+ 'Fold' + ' ' + 'ACC' + ' ' + 'AUC' + ' '+ '\n')
    save_evalustion = open(opt.saveFinalOut, 'a')
    save_evalustion.write(
        'stage'+' '+ 'ACC' + ' ' + 'AUC' + ' ' + 'SEN' + ' ' + 'SPE' + ' ' + 'F1-score' + ' ' + '\n')
    
    
    for fold in range(n_folds):
        print("\r\n************** in the Fold {}. ".format(fold))
        train_ind = cv_splits[fold][0] 
        test_ind = cv_splits[fold][1] 

        ### Constructing graphs for training and testing
        # extract node features  
        node_ftr = dl.get_node_features(train_ind)
        # get edge inputs
        edge_index, edgenet_input = dl.get_edge_inputs(nonimg)
        # normalization of edge inputs
        edgenet_input = (edgenet_input- edgenet_input.mean(axis=0)) / edgenet_input.std(axis=0)
        
        # build network architecture  
        model = DBwGcn(node_ftr.shape[1], opt.num_classes, opt.dropout, edge_dropout=opt.edropout, hgc=opt.hgc, lg=opt.lg, edgenet_input_dim=2*nonimg.shape[1],device=opt.device).to(opt.device)
        model = model.to(opt.device)

        # build loss, optimizer, metric 
        loss_fn =torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
        features_cuda = torch.tensor(node_ftr, dtype=torch.float32).to(opt.device)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
        edgenet_input = torch.tensor(edgenet_input, dtype=torch.float32).to(opt.device)
        labels = torch.tensor(y, dtype=torch.long).to(opt.device)
        fold_model_path = opt.ckpt_path + "/fold{}.pth".format(fold)


        def train():
            #Start training...
            acc = 0
            correct=0
            for epoch in range(opt.num_iter):
                model.train()
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    logit_de, logit_bys, node_logits = model(features_cuda, edge_index, edgenet_input)
                    l1 = loss_fn(logit_de[train_ind], labels[train_ind])
                    l2 = loss_fn(logit_bys[train_ind], labels[train_ind])
                    l12 = loss_fn(node_logits[train_ind], labels[train_ind])
                    loss = 0.25 * l1 + 0.25 * l2 + 0.5 * l12 + 0.000001
                    loss.backward()
                    optimizer.step()
                correct_train, acc_train = accuracy(node_logits[train_ind].detach().cpu().numpy(), y[train_ind])

                model.eval()
                with torch.set_grad_enabled(False):
                    logit_de,logit_bys,node_logits = model(features_cuda, edge_index, edgenet_input)
                logits_test = node_logits[test_ind].detach().cpu().numpy()
                correct_test, acc_test = accuracy(logits_test, y[test_ind])
                auc_test = auc(logits_test,y[test_ind])
                prf_test = prf(logits_test,y[test_ind])

                print("Epoch: {},\ttrain loss: {:.4f},\ttrain acc: {:.4f}".format(epoch, loss.item(), acc_train.item()))
                #print("Epoch: {},\teval loss: {:.4f},\teval acc: {:.4f}".format(epoch, loss_test.item(), acc_test))
                save_train_val.write( str(epoch) + ' ' + str(loss.item()) + ' ' + str(acc_train.item()) +  '\n')

                if acc_test > acc and epoch >9:
                    acc = acc_test
                    correct = correct_test
                    aucs[fold] = auc_test
                    prfs[fold]  = prf_test
                    if opt.ckpt_path !='':
                        if not os.path.exists(opt.ckpt_path):
                            os.makedirs(opt.ckpt_path)
                        torch.save(model.state_dict(), fold_model_path)

            accs[fold] = acc
            corrects[fold] = correct
            print("\r\n =>  Accuracy in {}-fold is {:.4f}".format(fold, acc))
            save_evalustion_folds.write('\n' + 'Train*' + ' '+ str(fold) + ' ' + str(acc) + '\n')

        def evaluate():
            #testing...
            model.load_state_dict(torch.load(fold_model_path))
            model.eval()
            logit_de,logit_bys,node_logits= model(features_cuda, edge_index, edgenet_input)

            logits_test = node_logits[test_ind].detach().cpu().numpy()
            corrects[fold], accs[fold] = accuracy(logits_test, y[test_ind])
            aucs[fold] = auc(logits_test,y[test_ind])
            prfs[fold]  = prf(logits_test,y[test_ind])
            print("  in fold {}, the test ACC {:.4f}, AUC {:.4f}".format(fold, accs[fold], aucs[fold]))
            save_evalustion_folds.write('\n' + 'Test*'+ ' ' + str(fold) + ' ' + str(accs[fold])+ ' ' + str(aucs[fold]) + '\n')


        if opt.train==1:
            train()
        elif opt.train==0:
            evaluate()



    n_samples = raw_features.shape[0]
    acc_nfold = np.sum(corrects)/n_samples
    print("=> Average test accuracy in {}-fold CV: {:.4f}".format(n_folds, acc_nfold))
    print("=> Average test AUC in {}-fold CV: {:.4f}".format(n_folds, np.mean(aucs)))
    se, sp, f1 = np.mean(prfs, axis=0)
    se_std, sp_std, f1_std = np.std(prfs, axis=0)
    print("=> Average test accuracy {:.4f}, AUC {:.4f}, sensitivity {:.4f}, specificity {:.4f}, F1-score {:.4f}".format(np.mean(accs),np.mean(aucs),se, sp, f1))
    save_evalustion.write(
        '\n' + 'Finally*' + ' ' + str(acc_nfold) + ' ' + str(np.mean(aucs)) + ' ' + str(se) + ' ' + str(sp) + ' ' + str(
            f1) + '\n')
    save_evalustion.write(
        'std' + ' ' + str(np.std(accs)) + ' ' + str(np.std(aucs)) + ' ' + str(se_std) + ' ' + str(sp_std) + ' ' + str(
            f1_std) + '\n')

    print('Congratulations! You have finished training or evaluating the model.')

