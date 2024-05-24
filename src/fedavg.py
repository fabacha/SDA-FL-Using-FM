import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, mal_inference, test_inference
from models import MLP, LeNet, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNFeMnist, CNNFeMnist_sim, CNNMiniImagenet, ConvNet
from utils import get_dataset, exp_details, average_weights, get_mal_dataset

if __name__ == '__main__':
    np.random.seed(903)
    torch.manual_seed(313) #cpu
    torch.cuda.manual_seed(322) #gpu
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups

    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

   
    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses, local_ns = [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        t_adv = False
        
        #rs_new_ep = []
        #rs_old_ep = []
        for idx in idxs_users:
            
            if args.dataset == 'HAR' or args.dataset == 'shakespeare' or 'extr_noniid' in args.dataset:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups_train[idx], logger=logger , dataset_test=test_dataset, idxs_test=user_groups_test[idx])
            else:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx], logger=logger, dataset_test=test_dataset)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, device=device)
            # get new model
            new_model = copy.deepcopy(global_model)
            new_model.load_state_dict(w)
            acc, _ = local_model.inference(model=new_model)
            print('user {}, loss {}, acc {}'.format(idx, loss, acc))
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_ns.append(1)
           
       

        #print(local_ns)
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)
        
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            print('Global model Test Accuracy: {:.2f}% \n'.format(100*test_acc))
           
    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))


    # Define the file path
    file_path = '../save/objects/'

    # Create the directory if it doesn't exist
    os.makedirs(file_path, exist_ok=True)

    # Construct the file name
    file_name = '{}{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.format(
        file_path, args.dataset, args.model, args.epochs, args.frac, args.iid,
        args.local_ep, args.local_bs
    )

    # Save the objects train_loss and train_accuracy
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)


    # # Saving the objects train_loss and train_accuracy:
    # file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)

    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)

    # print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

