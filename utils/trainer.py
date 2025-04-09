import json
import os
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

class Engine(object):
      """Engine that runs training and inference.
      Args
            - cur_epoch (int): Current epoch.
            - print_every (int): How frequently (# batches) to print loss.
            - validate_every (int): How frequently (# epochs) to run validation.
            
      """

      def __init__(self, model_, dataloader_train_, dataloader_val_, optimizer_, logdir_,
                  cur_epoch=0, cur_iter=0):
            self.model = model_
            self.dataloader_train = dataloader_train_
            self.dataloader_val = dataloader_val_
            self.cur_epoch = cur_epoch
            self.cur_iter = cur_iter
            self.bestval_epoch = cur_epoch
            self.train_loss = []
            self.val_loss = []
            self.bestval = 1e10

            self.optimizer = optimizer_

            self.writer = SummaryWriter(log_dir=logdir_)
            self.logdir = logdir_

      def train(self):
            loss_epoch = 0.
            num_batches = 0
            self.model.train()

            # Train loop
            for data in tqdm(self.dataloader_train):
                  
                  # efficiently zero gradients
                  for p in self.model.parameters():
                        p.grad = None
                  
                  if isinstance(self.model, nn.DataParallel):
                        data_in, label = self.model.module.data_preprocess(data)
                  else :
                        data_in, label = self.model.data_preprocess(data)

                  pred = self.model(data_in)

                  if isinstance(self.model, nn.DataParallel):
                        loss = self.model.module.get_loss(pred, label)
                  else:
                        loss = self.model.get_loss(pred, label)
                        
                  loss.backward()
                  loss_epoch += float(loss.item())
                  num_batches += 1

                  self.optimizer.step()

                  self.writer.add_scalar('train_loss', loss.item(), self.cur_iter)
                  self.writer.add_scalar('lr', self.optimizer.lr, self.cur_iter)

                  self.cur_iter += 1
            
            loss_epoch = loss_epoch / num_batches
            self.train_loss.append(loss_epoch)
            self.cur_epoch += 1

      def validate(self):
            self.model.eval()

            with torch.no_grad():	
                  num_batches = 0
                  wp_epoch = 0.

                  # Validation loop
                  for batch_num, data in enumerate(tqdm(self.dataloader_val), 0):
                        
                        if isinstance(self.model, nn.DataParallel):
                              data_in, label = self.model.module.data_preprocess(data)
                        else :
                              data_in, label = self.model.data_preprocess(data)
                  
                        pred = self.model(data_in)

                        if isinstance(self.model, nn.DataParallel):
                              loss = self.model.module.get_loss(pred, label)
                        else:
                              loss = self.model.get_loss(pred, label)

                        # gt_waypoints = [torch.stack(data['waypoints'][i], dim=1).to(args.device, dtype=torch.float32) for i in range(config.seq_len, len(data['waypoints']))]
                        # gt_waypoints = torch.stack(gt_waypoints, dim=1).to(args.device, dtype=torch.float32)
                        wp_epoch += float(loss)

                        num_batches += 1
                              
                  wp_loss = wp_epoch / float(num_batches)
                  tqdm.write(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:' + f' loss: {wp_loss:3.3f}')

                  self.writer.add_scalar('val_loss', wp_loss, self.cur_epoch)
                  
                  self.val_loss.append(wp_loss)

      def save(self):

            save_best = False
            if self.val_loss[-1] <= self.bestval:
                  self.bestval = self.val_loss[-1]
                  self.bestval_epoch = self.cur_epoch
                  save_best = True
                  
            # Create a dictionary of all data to save
            log_table = {
                  'epoch': self.cur_epoch,
                  'iter': self.cur_iter,
                  'bestval': self.bestval,
                  'bestval_epoch': self.bestval_epoch,
                  'train_loss': self.train_loss,
                  'val_loss': self.val_loss,
            }

            # Save ckpt for every epoch
            if isinstance(self.model, nn.DataParallel):
                  torch.save(self.model.module.state_dict(), os.path.join(self.logdir, 'model_%d.pth'%self.cur_epoch))
            else:
                  torch.save(self.model.state_dict(), os.path.join(self.logdir, 'model_%d.pth'%self.cur_epoch))

            # Save the recent model/optimizer states
            if isinstance(self.model, nn.DataParallel):
                  torch.save(self.model.module.state_dict(), os.path.join(self.logdir, 'model.pth'))
            else:
                  torch.save(self.model.state_dict(), os.path.join(self.logdir, 'model.pth'))
            # torch.save(self.optimizer.state_dict(), os.path.join(self.logdir, 'recent_optim.pth'))

            # Log other data corresponding to the recent model
            with open(os.path.join(self.logdir, 'recent.log'), 'w') as f:
                  f.write(json.dumps(log_table))

            tqdm.write('====== Saved recent model ======>')
            
            if save_best:
                  if isinstance(self.model, nn.DataParallel):
                        torch.save(self.model.module.state_dict(), os.path.join(self.logdir, 'best_model.pth'))
                  else:
                        torch.save(self.model.state_dict(), os.path.join(self.logdir, 'best_model.pth'))
                  # torch.save(self.optimizer.state_dict(), os.path.join(self.logdir, 'best_optim.pth'))
                  tqdm.write('====== Overwrote best model ======>')

