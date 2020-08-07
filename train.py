from __future__ import absolute_import, division, print_function, unicode_literals
import argparse, random, os, logging, glob
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import AdamW, WarmupLinearSchedule
import torch_optimizer as optim

import utils.constant as config
from utils.data_loader import NERDataset, pad_collate
from utils.log import logger, init_logger
from models.cnn_lstm import CNNBiLSTM
device = config.device


def to_cpu(tensors):
    for t in tensors:
        t.cpu()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-epoch", default=config.epoch, type=int)
    parser.add_argument("-batch_size", default=config.batch_size, type=int)
    parser.add_argument("-lstm_layers", default=config.lstm_layers, type=int)
    parser.add_argument("-channel_in", default=config.channel_in, type=int)
    parser.add_argument("-channel_out", default=config.channel_out, type=int)
    parser.add_argument("-kernel_sizes", default=config.kernel_sizes, type=list)
    parser.add_argument("-max_char_len", default=config.max_char_len, type=int)
    parser.add_argument('-save_dir', default='/epoch_{}_batch_{}_ch_in_{}_ch_out_{}')
    # parser.add_argument("-is_distill", type=str2bool, nargs='?',const=True,default=False)
    
    args = parser.parse_args()
    log_path = './result' + args.save_dir.format(args.epoch, args.batch_size, args.channel_in, args.channel_out,)
    init_logger(log_path,'/log.txt')
    tb_writer = SummaryWriter('{}/runs'.format(log_path))

    # Load Entity Dictionary, Train and Test data
    vocab_size = len(torch.load('./data/word_vocab.pt'))
    char_vocab_size = len(torch.load('./data/char_vocab.pt'))
    pos_vocab_size = len(torch.load('./data/pos_vocab.pt'))
    entitiy_to_index = torch.load('./data/processed_data/entity_to_index.pt')
    num_class = len(entitiy_to_index)

    print("Load processed data...")
    # Load process train and validation data
    train_dataset = NERDataset('tr')
    valid_dataset = NERDataset('valid')

    # Build train_and validation loaders which generate data with batch_size
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=pad_collate, drop_last=True, num_workers=0)

    valid_loader = DataLoader(dataset=valid_dataset,batch_size=args.batch_size, shuffle=True,
                              collate_fn=pad_collate, drop_last = True, num_workers=0)
    
    print("Build Model...")
    model = CNNBiLSTM(config, num_class, vocab_size, char_vocab_size, pos_vocab_size)
    criterion = nn.CrossEntropyLoss()
    
    # Prepare optimizer and schedule (linear warmup and decay)
    print("Set Optimized and Scheduler...")
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}]
    
    # t_total = train_examples_len // model_config.gradient_accumulation_steps * model_config.epochs
    t_total = len(train_loader) * args.epoch
    # optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    optimizer = optim.RAdam(model.parameters(), lr=config.learning_rate)
    # scheduler = WarmupLinearSchedule(optimizer, int(t_total*0.1), t_total)
     
    # Train
    logger.info("***** Running training *****")
    logger.info("  Num train examples = %d", len(train_loader))
    logger.info("  Num validation examples = %d", len(valid_loader))
    logger.info("  Num Epochs = %d", args.epoch)
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Total steps = %d", t_total)
    logger.info("  Model save directory = %s", log_path)
    
    global_step = 0
    best_eval_acc = 0.0
    

    print('Train Start !')
    for e in tqdm(range(args.epoch)):
        tr_acc, tr_loss = 0.0, 0.0
        
        model.train()
        model.to(device)
        # print('Train at Epoch {}'.format(e+1))
        for step, batch in enumerate(train_loader):
            global_step += 1
            token_idx, char_idx, pos_idx, label = batch
            
            # Model Input
            logit = model(token_idx, char_idx, pos_idx)
            loss = criterion(logit.view(-1, logit.size(-1)), label.view(-1))
            tr_loss += loss.item()
            loss.backward()
            
            with torch.no_grad():
                accuracy = (logit.argmax(-1)==label).float()[label!=0].mean().item()
                tr_acc += accuracy
                
            to_cpu([token_idx, char_idx, pos_idx, label])
        
            if global_step >0 and global_step % config.gradient_accumulation_steps == 0:    
                # global_step += config.gradient_accumulation_steps                
                optimizer.step()
                optimizer.zero_grad()
                model.zero_grad()
                # scheduler.step()
                
#             if global_step % 100==0: #int(len(self.train_dataloader)/5) ==0:
#                 tr_avg_acc = tr_acc / global_step
#                 tr_avg_loss = tr_loss / global_step

#                 logger.info('epoch : {} /{}, global_step : {} /{}, tr_avg_loss: {:.5f}, tr_loss : {:.5f}, tr_avg_acc: {:.2%}, tr_acc: {:.2%}'.format(
#                     e+1, args.epoch, global_step, t_total, tr_avg_loss, loss.item(), tr_avg_acc, accuracy))
#                 tb_writer.add_scalars('tr_loss', {'average': tr_avg_loss, 'current': loss.item()}, global_step)
        tr_avg_acc = tr_acc / step
        tr_avg_loss = tr_loss / step
        logger.info('>>> Train Epoch : {}  Tr_avg_loss: {:.5f}, Tr_avg_acc: {:.2%}'.format(e+1, tr_avg_loss, tr_avg_acc))        
        
        """ Evaluation"""
                
        model.eval()
        eval_acc, eval_loss = 0.0, 0.0
        # print('Evaluate at Epoch {}'.format(e+1))
        
        for step, batch in enumerate(valid_loader):
            token_idx, char_idx, pos_idx, label = batch
            logit = model(token_idx, char_idx, pos_idx)
            loss = criterion(logit.view(-1, logit.size(-1)), label.view(-1))
            eval_loss += loss.item()
            
            with torch.no_grad():
                eval_accuracy = (logit.argmax(-1)==label).float()[label!=0].mean().item()
                eval_acc += eval_accuracy
                
        to_cpu([token_idx, char_idx, pos_idx, label])
        eval_avg_acc = eval_acc / step
        eval_avg_loss = eval_loss / step
        
        logger.info('VALID epoch : {} /{}, Eval_avg_loss: {:.5f}, Eval_avg_acc: {:.2%}'.format(e+1, args.epoch, eval_avg_loss, eval_avg_acc))    
        tb_writer.add_scalars('eval_loss', {'average': eval_avg_loss, 'current': loss.item()}, global_step)
        
        """"""

        # Model Save
        if eval_avg_acc > best_eval_acc:
            model.to('cpu')
            best_eval_acc = eval_avg_acc
            state = {'epoch':e+1,'model_state_dict': model.state_dict()}
            save_path = '{}/epoch_{}_step_{}_tr_acc_{:.3f}_eval_acc_{:.3f}.pt'.format(
                        log_path, e+1, global_step, tr_avg_acc, eval_avg_acc)
            
            if len(glob.glob(log_path+'/epoch*.pt'))>0:
                os.remove(glob.glob(log_path+'/epoch*.pt')[0])

            torch.save(state, save_path)
            logger.info('Model saving with best eval acc : {:.2%}'.format(eval_avg_acc))
            
            # if len(glob.glob(log_path+'/epoch*.pt'))==0:
            os.mkdir(log_path+'/epoch_{}_step_{}_tr_acc_{:.3f}_eval_acc_{:.3f}'.format(e+1, global_step, tr_avg_acc, eval_avg_acc))