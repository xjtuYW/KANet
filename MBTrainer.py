import os
from tqdm import tqdm
import torch.nn.functional as F

from tools.utils import *
from models import KPM
from models.MyModel_CLIP import MyModel

from torch.utils.tensorboard import SummaryWriter

import pdb

"""
# note: the parameters of self.mode.fc are updated by the encoded feature, 
#       thus we treat the parameter as general knowledge.

#       the parameters of self.mode.heads_vis are learned from differenet classes, 
#       thus we treat the parameter as class specific knowledge. 

if use text info, then there are two choices, using text prompts or not
if use visul info, then the only choice is whether use prompt

"""

class MBTrainer():
    def __init__(self, args):
        self.args       = args
        self.device     = "cuda" if torch.cuda.is_available() else "cpu"
        self.inc_shot   = 5  
        # init paths
        self.cwd = os.getcwd()
        if self.args.DLC:
            self.cwd = self.cwd.replace('/root/code/fscil', '/root/data/wy/FSIL') 
        self.init_paths()
        print("Current work dir is {}".format(self.work_dir))
        check_dir(self.work_dir)
        # initi incremental info
        self.model = MyModel(dataset=self.args.dataset,
                            arch_name=self.args.network,
                            prompt_len=self.args.prompt_len).to(self.device)

        self.KPM   = KPM(self.model.hdim).to(self.device)
        self.model_save_name        = 'MB.pth' #
        self.model_save_key         = 'MB' # 
        self.best_model_save_path   = None

        # performance records
        self.performance_stati = {
            'details': torch.zeros(self.model.sessions, self.model.sessions), # acc on each session
            'forgetting': 0,
            'acc_base': 0,
            'acc_novel': 0,
            'acc_each_sess': torch.zeros(self.model.sessions)
        }

        # summary writer
        self.writer = SummaryWriter(os.path.join(self.work_dir, 'tensorboard'))

        # global variables
        self.memory = None

        # global info 
        if self.args.full_data:
            print("Class-Incremental Learning")
        else:
            print("Few-Shot Class-Incremental Learning")

        if self.args.cap_layer != -1:
            print("Memory activated!")
        if self.args.enable_vis:
            print("Image branch activated......")
        if self.args.enable_prompt:
            print("method build on Pro-T, the len of prompt is {}".format(self.args.prompt_len))
          
    def init_paths(self):
        # self.dir_1 = '{self.args.parent_folder}'
        
        if self.args.dataset == 'ImageNet_R':  # 
            self.data_prefix            = 'imr_' 
        elif self.args.dataset == 'cifar100':
            self.data_prefix            = 'cifar_' 
        elif self.args.dataset == 'cub_200':
            self.data_prefix                 = 'cub_'
        elif self.args.dataset == 'miniImageNet':
            self.data_prefix                 = 'mini_'  
        elif self.args.dataset == 'mnist':
            self.data_prefix                 = 'mnist_'
        elif self.args.dataset == 'flowers':
            self.data_prefix                 = 'flowers_'  
        elif self.args.dataset == 'food101':
            self.data_prefix                 = 'food101_'
        elif self.args.dataset == 'car196':
            self.data_prefix                 = 'car196_'   
        elif self.args.dataset == 'aircraft102':
            self.data_prefix                 = 'aircraft102_'
        elif self.args.dataset == 'cifar10':
            self.data_prefix                 = 'cifar10_'
        else:
            raise ValueError(f"Invalid dataset name {self.args.dataset}")

        self.work_dir = os.path.join(self.cwd, f'experiments/{self.args.parent_folder}/{self.args.dataset}/'+self.args.storage_folder)  # 
        # examplars
        self.examplar_num_per_cls = 5
                

    def get_train_loader(self, 
                         sampler_type: str='std', 
                         joint: bool=False, 
                         transform_state: str='train',
                         full_data: bool=False,
                         cur_session: int=0):
        """
        @ sampler_type: sample batch if 'std' or sample task  if 'fsl'
        """
        if joint:
            self.args.used_data = self.data_prefix+'joint'
        else:
            self.args.used_data = self.data_prefix+'base'

        if full_data:
            self.args.used_data = self.data_prefix+'full'

        self.args.state     = transform_state
        self.args.sampler   = sampler_type

        min_cls = 0 if (cur_session == 0 or joint) else self.model.base_cls_num + (cur_session-1) * self.model.inc_cls_num
        max_cls = self.model.base_cls_num + cur_session * self.model.inc_cls_num
        if sampler_type == 'fsl':
            sample_info=[self.args.tasks, 1, self.args.n_way, self.args.n_shot, self.args.n_query, max_cls, min_cls]
            train_loader = get_dataloader(self.args, dataset_min_cls=min_cls, dataset_max_cls=max_cls, sample_info=sample_info)
        elif sampler_type == 'align_inc':
            sample_info=[self.args.tasks, 1, self.args.n_way, self.args.n_shot, self.args.n_query, max_cls, min_cls, self.args.batch_size]
            train_loader = get_dataloader(self.args, dataset_min_cls=min_cls, dataset_max_cls=max_cls, sample_info=sample_info)
        else:
            train_loader = get_dataloader(self.args, dataset_min_cls=min_cls, dataset_max_cls=max_cls)

        return train_loader
    

    def get_inc_loader(self, ways: int=10,  shots: int=5, max_cls: int=10000, min_cls:int=0):
        self.args.state = 'test'
        if shots > 5:
            self.args.used_data = self.data_prefix + 'full'
        else:
            self.args.used_data = self.data_prefix + 'inc'
        self.args.sampler = 'inc'
        inc_train_loader = get_dataloader(self.args, 
                                          dataset_min_cls= min_cls, 
                                          dataset_max_cls= max_cls,
                                          sample_info=[ways, shots])

        return inc_train_loader


    def get_test_loader(self, num_seen_cls):
        self.args.state = 'test'
        self.args.used_data = self.data_prefix+'test'
        self.args.sampler = 'std'
        test_loader = get_dataloader(self.args, dataset_max_cls=num_seen_cls, sample_info=[self.args.batch_size_test, num_seen_cls])
        
        return test_loader


    def construct_memory(self, cur_session: int=0):
        memory = self.get_memory(cur_session=cur_session)
        if cur_session != 0:
            memory       = torch.cat((self.memory, memory), dim=0)
        print(f"the memory size is {memory.shape}")
        return memory


    def calc_logits_loss(self, 
                proto, 
                que_feat, 
                mode='train'):
        logits = 0.0
        visual_logits = F.linear(F.normalize(que_feat, p=2, dim=-1), F.normalize(proto, p=2, dim=-1)) # [n_q, n_w]
        logits        += F.softmax(self.args.temperature * visual_logits, dim=-1)

        # get losses
        if mode == 'train':
            # relative labels
            rel_label = torch.arange(self.args.n_way).repeat(self.args.n_query).type(torch.cuda.LongTensor)
            loss  = F.cross_entropy(self.args.temperature * visual_logits, rel_label)
            acc = count_accuracy(logits, rel_label)
            return loss, acc
        else:
            return logits


    def train(self, 
              load_pretrained=True,
              cur_session: int=0):
        """
        @ para 
        """
        # initialize relevant objects
        train_log = os.path.join(self.work_dir, 'train.log')
        log(train_log, str(vars(self.args)))
        model_save_path = os.path.join(self.work_dir, self.model_save_name)
        self.best_model_save_path = model_save_path
        n_w, n_s, n_q = self.args.n_way, self.args.n_shot, self.args.n_query
        num_sup       = n_w * n_s

        self.model.eval()

        # construct memory
        if self.args.cap_layer != -1 and self.args.version=='V1':
            self.memory = self.construct_memory(cur_session=cur_session)
            memory_tmp  = self.memory.clone().detach()

        # few-shot train loader
        train_loader = self.get_train_loader(sampler_type='align_inc',
                                            cur_session=cur_session, 
                                            full_data=self.args.full_data)
        # initilize optimizer
        optimizer, scheduler = self.get_optim('task_train')

        # training
        timer = Timer()
        max_val_acc = 0.0
        train_loss, train_loss_base, train_loss_inc, train_loss_global = [], [], [], []
        writer_axis_x = 0
        for epoch in range(self.args.epoch):
            self.memory = memory_tmp.clone().detach()
            log(train_log, 'Session:{}\tTrain Epoch:{}\tLearning Rate:{:.6f}'.format(0, epoch, scheduler.get_last_lr()[0]))
            self.KPM.train()
            for i, batch in enumerate(tqdm(train_loader)):
                # split data into support and query
                data, true_label = [_.cuda() for _ in batch] 
                loss = 0
                    
                inc_data_num          = n_w * (n_s + n_q)
                inc_data, base_data   = data[:inc_data_num], data[inc_data_num:]
                inc_label, base_label = true_label[:inc_data_num], true_label[inc_data_num:]

                sup_data, que_data    = inc_data[:num_sup], inc_data[num_sup:] 
                sup_label, que_label  = inc_label[:num_sup], inc_label[num_sup:]

                data_old_new          = torch.cat((base_data, que_data), dim=0)
                label_old_new         = torch.cat((base_label, que_label), dim=0)
                
                # unique inc classes
                uni_label_inc          = self.get_unique_label(sup_label)
                idx_inc                = [x.item() for x in uni_label_inc]

                # memory 
                keep_idx = []
                for ind in range(self.model.base_cls_num):
                    if ind not in idx_inc:
                        keep_idx.append(ind)
                memory_    = self.memory.clone().reshape(-1, self.examplar_num_per_cls, 768)
                memory_old = memory_[keep_idx].reshape(-1, 768)

                with torch.no_grad():
                    sup_feats_  = self.model.encode_image(sup_data, enable_prompt=self.args.enable_prompt, cap_layer=self.args.cap_layer)
                    memory_new = sup_feats_[:, 0, :].reshape(n_s, n_w, -1, 768).transpose(1, 0).mean(dim=1)
                    memory_new = memory_new.reshape(-1, 768)
                    memory_old_new  = torch.cat((memory_old, memory_new), dim=0)

                # loss on old classes
                base_feats      = self.model.encode_image(base_data,  
                                                            memory=memory_old, 
                                                            KPM=self.KPM, 
                                                            upd_layer=self.args.upd_layer,
                                                            upd_targt=self.args.upd_targt,
                                                            enable_prompt=self.args.enable_prompt)
                base_logits     = F.linear(F.normalize(base_feats, p=2, dim=-1), 
                                            F.normalize(self.model.heads_vis[0].weight[keep_idx], p=2, dim=-1)) 
                
                label_map       = dict(zip(keep_idx, range(len(keep_idx))))
                mapped_labels   = [label_map[x.item()] for x in base_label]
                mapped_labels   = torch.LongTensor(mapped_labels).to(self.device)
                loss_base       = F.cross_entropy(self.args.temperature * base_logits, mapped_labels)
                loss            += self.args.loss_base * loss_base

                # loss on inc classes
                sup_feats       = self.model.encode_image(sup_data,  
                                                        memory=memory_new, 
                                                        KPM=self.KPM, 
                                                        upd_layer=self.args.upd_layer,
                                                        upd_targt=self.args.upd_targt,
                                                        enable_prompt=self.args.enable_prompt)

                que_feats       = self.model.encode_image(que_data, 
                                                        memory=memory_new, 
                                                        KPM=self.KPM, 
                                                        upd_layer=self.args.upd_layer,
                                                        upd_targt=self.args.upd_targt,
                                                        enable_prompt=self.args.enable_prompt)
                proto           = sup_feats.reshape(n_s, n_w, -1).mean(dim=0)
                loss_inc, acc   = self.calc_logits_loss(proto, que_feats, seman=None, que_label=que_label, memory=memory_new)
                loss            += self.args.loss_inc * loss_inc

                # global loss
                feat_old_new    = self.model.encode_image(data_old_new, 
                                                        memory=memory_old_new, 
                                                        KPM=self.KPM, 
                                                        upd_layer=self.args.upd_layer,
                                                        upd_targt=self.args.upd_targt,
                                                        enable_prompt=self.args.enable_prompt)
                # 
                rel_label_inc   = torch.arange(self.args.n_way).repeat(self.args.n_query).type(torch.cuda.LongTensor)
                rel_label_inc   += self.model.base_cls_num - self.args.n_way
                rel_label_base  = mapped_labels
                label_old_new   = torch.cat((rel_label_base, rel_label_inc), dim=0)
                global_weights  = torch.cat((self.model.heads_vis[0].weight[keep_idx], proto), dim=0)
                global_logits   = F.linear(F.normalize(feat_old_new, p=2, dim=-1), F.normalize(global_weights, p=2, dim=-1)) 

                loss_global     = F.cross_entropy(self.args.temperature * global_logits, label_old_new)
                loss            += self.args.loss_global * loss_global

                # tmp results
                train_loss.append(loss.item())
                train_loss_inc.append(loss_inc.item())
                train_loss_global.append(loss_global.item())
                train_loss_base.append(loss_base.item())

                # update paras
                optimizer.zero_grad();loss.backward();optimizer.step()

                # output log
                if i % 10 == 0:
                    train_loss_avg  = np.mean(np.array(train_loss))
                    log(train_log, 'Train Epoch:{}\tBatch:[{}/{}]\tLoss:{:.4f} % ({:.4f})'.format(
                            epoch, i, len(train_loader), train_loss_avg, loss.item()))
                
                self.writer.add_scalar('loss', np.mean(np.array(train_loss)), writer_axis_x)
                self.writer.add_scalar('loss_global', np.mean(np.array(train_loss_global)), writer_axis_x)
                self.writer.add_scalar('loss_inc', np.mean(np.array(train_loss_inc)), writer_axis_x)
                self.writer.add_scalar('loss_base', np.mean(np.array(train_loss_base)), writer_axis_x)
                writer_axis_x += 1
            scheduler.step()

            # save model
            if epoch % 1 == 0 and cur_session == 0:
                val_acc_avg = self.test(reload=False, base_session=True)
                log(train_log, 'Validation Epoch:{}\tAccuracy:{:.2f}'.format(epoch, val_acc_avg))
                if val_acc_avg > max_val_acc:
                    max_val_acc = val_acc_avg
                    torch.save(
                    {self.model_save_key: self.KPM.state_dict(), 'base_model': self.model.state_dict()}, 
                    model_save_path)  

                log(train_log, 'Elapsed Time: {}/{}\n'.format(
                                                timer.measure(), timer.measure((epoch+1) / float(self.args.epoch+1))))
        return 0


    def finetuning(self, data, label, cur_session, epochs: int=5):
        # set optimizer
        optimizer, scheduler =  self.get_optim('pretrain', initial_lr=0.003)
        self.model.eval()
        for epoch in range(epochs):
            visual_emb  = self.model.encode_image(data, enable_prompt=self.args.enable_prompt)
            logits = []
            for j in range(cur_session+1):
                logits.append(
                    F.linear(
                        F.normalize(visual_emb, p=2, dim=-1), F.normalize(self.model.heads_vis[j].weight, p=2, dim=-1)
                        ))
            logits = torch.cat(logits, dim=1)
            loss   = F.cross_entropy(self.args.temperature * logits, label)
            print(f"loss:{loss.item()}")
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        

    @torch.no_grad()
    def single_session_test(self, cur_session: int=0, memory=None, mode: str='inc', weight_update_mode: str='init'):

        # 
        num_seen_cls = self.model.base_cls_num + cur_session * self.model.inc_cls_num
        test_accs, collect_preds, collect_labels = [], [], []


        # get test_loader
        inc_test_loader = self.get_test_loader(num_seen_cls)
        for i, batch in enumerate(tqdm(inc_test_loader)):
            data, label = [_.cuda() for _ in batch]
            visual_emb  = self.model.encode_image(data, 
                                                memory=memory, 
                                                KPM=self.KPM, 
                                                upd_layer=self.args.upd_layer,
                                                upd_targt=self.args.upd_targt,
                                                enable_prompt=self.args.enable_prompt)
            protos  = self.model.protos.weight.data[:num_seen_cls]
            logits  = self.calc_logits_loss(protos, visual_emb, mode='test')

            # calc acc
            preds    = torch.argmax(torch.softmax(logits, dim=-1), dim=-1).reshape(-1)
            accuracy = 100 * preds.eq(label).float().mean()
            test_accs.append(accuracy.item())
            collect_preds.append(preds)
            collect_labels.append(label)
           
        # performance on each session
        collect_preds   = torch.cat(collect_preds, dim=0)
        collect_labels  = torch.cat(collect_labels, dim=0)

        if self.model.inc_cls_num != 1:
            self.performance_analysis(cur_session, collect_preds, collect_labels)
        return np.mean(np.array(test_accs))


    def test(self, reload: bool=False, mode: str='inc', base_session: bool=False, weight_update_mode: str='init', align_first_session:bool=False):

        test_log = os.path.join(self.work_dir, 'test_' +'.log')
        if reload:
            self.KPM, self.model = load_trained_paras(os.path.join(self.work_dir, self.model_save_name), 
                                                        [self.KPM, self.model], [self.model_save_key, 'base_model'])
        self.model.eval();self.KPM.eval()

        # construct memory
        if self.memory is None:
            self.memory = self.construct_memory(cur_session=0)

        # acc = self.single_session_test(0, memory=self.memory, =mode=mode, weight_update_mode=weight_update_mode)

        # in the first session, we do not use the matching method
        if align_first_session:
            acc = self.single_session_test(0, memory=self.memory, mode='pretrain', weight_update_mode='original')
            self.update_fc(memory=self.memory)
        else:
            self.update_fc(memory=self.memory)
            acc = self.single_session_test(0, memory=self.memory, mode=mode, weight_update_mode=weight_update_mode)

        self.performance_stati['acc_each_sess'][0] = acc
        log(test_log, 'Sess:{}\tAcc:{:.2f}'.format(0, acc))
        
        # session > 1
        if not base_session:
            if self.args.full_data:
                for sess in range(self.model.sessions-1):
                    self.args.epoch = 0
                    self.args.lr = 0.0001
                    self.args.n_way = self.model.inc_cls_num
                    self.args.tasks = 10
                    # self.train(cur_session=sess+1, load_pretrained=False)
                    self.update_fc(memory=self.memory, cur_session=sess+1)
                    acc = self.single_session_test(sess+1, memory=self.memory, mode=mode, weight_update_mode=weight_update_mode)
                    self.performance_stati['acc_each_sess'][sess+1] = acc
                    log(test_log, 'Sess:{}\tAcc:{:.2f}'.format(sess+1, acc))
            else:
                inc_train_loader = self.get_inc_loader(self.model.inc_cls_num, 
                                                       self.inc_shot, 
                                                       max_cls=self.model.num_cls,
                                                       min_cls=self.model.base_cls_num)
                for sess, batch in enumerate(tqdm(inc_train_loader)):
                    data, label     = [_.cuda() for _ in batch]
                    if self.args.mode != 'pretrain' and self.args.cap_layer != -1:
                        self.memory = self.construct_memory(cur_session=sess+1)

                    if weight_update_mode == 'init':
                        self.update_inc_fc(data, label, cur_session=sess+1, memory=self.memory)
                    else:
                        self.finetuning(data, label, sess+1)

                    acc = self.single_session_test(sess+1, memory=self.memory, mode=mode, weight_update_mode=weight_update_mode)
                    self.performance_stati['acc_each_sess'][sess+1] = acc
                    log(test_log, 'Sess:{}\tAcc:{:.2f}'.format(sess+1, acc))
        
        log(test_log, 'Performance info:{}'.format(self.performance_stati))
        log(test_log, 'Average: {}'.format(self.performance_stati['acc_each_sess'].mean()))
        return acc


    def performance_analysis(self, current_session: int, collect_preds: torch.tensor, collect_labels: torch.tensor):
        for i in range(current_session+1):
            if i == 0:
                start_ = 0
                end_   = self.model.base_cls_num
            else:
                start_ = self.model.base_cls_num + (i-1) * self.model.inc_cls_num
                end_   = self.model.base_cls_num + i * self.model.inc_cls_num
            
            # label belong to [start_, end_)
            idx_lt = torch.lt(collect_labels, end_).nonzero(as_tuple=False).squeeze()
            idx_ge = torch.ge(collect_labels, start_).nonzero(as_tuple=False).squeeze()
            idx    = [j for j in idx_ge if j in idx_lt]

            select_preds  = collect_preds[idx]
            select_labels = collect_labels[idx]

            # calc acc
            cur_acc = 100 * select_preds.eq(select_labels).float().mean()
            self.performance_stati['details'][i][current_session] = cur_acc
        
        # last session
        if current_session == self.model.sessions - 1:
            details                 = self.performance_stati['details']
            # forgetting
            performance_prev        = details[:-1, :-1] # k-1 step
            performance_prev_max, _ = torch.max(performance_prev, dim=-1)
            performance_last        = details[:-1, -1]  # k step
            forgetting              = torch.mean(performance_prev_max-performance_last).item()

            # acc base and acc novel
            last_base               = details[0, -1].item()
            acc_novel               = torch.mean(details[1:, -1]).item()

            self.performance_stati['forgetting']    = forgetting
            self.performance_stati['acc_base']      = last_base
            self.performance_stati['acc_novel']     = acc_novel


    def get_optim(self, mode: str='pretrain', initial_lr: float=-1):
        # set optim target
        if initial_lr == -1:
            initial_lr = self.args.lr
        if mode == 'pretrain':
            optim_target = [{'params': self.model.parameters(),'lr':initial_lr}]
            if 'ViT' in self.args.network:
                for p in self.model.encoder.parameters():
                    p.requires_grad = False
        else:

            optim_target = [{'params': self.KPM.parameters(),'lr':initial_lr}]

            if 'ViT' in self.args.network:
                for p in self.model.encoder.parameters():
                    p.requires_grad = False
          

        # select optimizer
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(optim_target, momentum=self.args.momentum, weight_decay=self.args.wd, nesterov=self.args.nesterov)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(optim_target, weight_decay=self.args.wd)
        else:
            raise ValueError(f"Invalid optimizer {self.args.optimizer}")

        # select scheduler
        if self.args.scheduler == 'SLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                                    optimizer, step_size=self.args.steps, gamma=self.args.gamma)
        elif self.args.scheduler == 'MSLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                    optimizer, milestones=self.args.milestones, gamma=self.args.gamma)
        elif self.args.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.epoch)
        else:
            raise ValueError(f"Invalid scheduler {self.args.scheduler}")
        
        return optimizer, scheduler


    @torch.no_grad()
    def update_fc(self, memory=None, base_update_mode: str='init', cur_session: int=0):
        min_cls = 0 if cur_session ==0 else self.model.base_cls_num + (cur_session-1) * self.model.inc_cls_num
        max_cls = self.model.base_cls_num + cur_session * self.model.inc_cls_num

        self.model.eval()
        if cur_session == 0:
            self.args.state     = 'test'
            self.args.used_data = self.data_prefix + 'base'
            self.args.sampler   = 'std'
            data_loader         = get_dataloader(self.args)
        else:
            data_loader = self.get_train_loader(transform_state='test',
                                                full_data=self.args.full_data,
                                                cur_session=cur_session)
        embedding_list      = []
        label_list          = []

        # get embbeddings
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader)):
                data, label = [_.cuda() for _ in batch]
                embedding   = self.model.encode_image(data, 
                                                    memory=memory, 
                                                    KPM=self.KPM, 
                                                    upd_layer=self.args.upd_layer,
                                                    upd_targt=self.args.upd_targt,
                                                    enable_prompt=self.args.enable_prompt)

                embedding_list.append(embedding.detach().cpu())
                label_list.append(label)
        embedding_list      = torch.cat(embedding_list, dim=0).to(self.device)
        # get prototypes
        label_list          = torch.cat(label_list, dim=0)
        proto_list          = []
        for class_index in range(min_cls, max_cls):
            data_index      = (label_list == class_index).nonzero(as_tuple=False)
            embedding_this  = embedding_list[data_index.squeeze(-1)] # [N, self.emb_dim]
            embedding_this  = embedding_this.mean(0)
            proto_list.append(embedding_this.squeeze())
        proto_list = torch.stack(proto_list, dim=0)

        # save protos
        self.model.protos.weight.data[min_cls:max_cls]     = proto_list 
        self.model.heads_vis[cur_session].weight.data      = proto_list
        

    @torch.no_grad()
    def update_inc_fc(self, 
                      data=None, 
                      label=None, 
                      memory=None, 
                      cur_session: int=1):
        start_label = self.model.base_cls_num + (cur_session-1) * self.model.inc_cls_num
        end_label   = self.model.base_cls_num + cur_session * self.model.inc_cls_num
        cls_list    = np.arange(start_label, end_label)
        
        data = self.model.encode_image(data, 
                                       memory=memory, 
                                       KPM=self.KPM, 
                                       upd_layer=self.args.upd_layer,
                                       upd_targt=self.args.upd_targt,
                                       enable_prompt=self.args.enable_prompt)      

        for class_index in cls_list:
            data_index = (label==class_index).nonzero(as_tuple=False).squeeze(-1)
            embedding = data[data_index]
            proto = embedding.mean(0)
            self.model.protos.weight.data[class_index] = proto


    def get_unique_label(self, label):
        uni_label = []
        for l in label:
            if l not in uni_label:
                uni_label.append(l)
        return uni_label


    @torch.no_grad()
    def get_memory(self, cur_session: int=0):
        self.model.eval()
        num_seen_cls = self.model.base_cls_num + cur_session * self.model.inc_cls_num
        start_idx    = 0 if cur_session == 0 else num_seen_cls - self.model.inc_cls_num

        # # get dataloder
        if cur_session == 0:
            self.args.state     = 'test'
            self.args.used_data = self.data_prefix + 'base'
            self.args.sampler   = 'std'
            dataloader          = get_dataloader(self.args)
        elif self.args.full_data:
            dataloader = self.get_train_loader(transform_state='test', 
                                               full_data=self.args.full_data, 
                                               cur_session=cur_session)
        else:
            max_cls    = self.model.base_cls_num + cur_session * self.model.inc_cls_num
            min_cls    = max_cls - self.model.inc_cls_num
            dataloader = self.get_inc_loader(self.model.inc_cls_num, self.inc_shot, max_cls, min_cls)
        
        class_embedding_list      = []
        label_list                = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader)):
                data, label = [_.cuda() for _ in batch]
                embeddings = self.model.encode_image(data, 
                                                     cap_layer=self.args.cap_layer,
                                                     enable_prompt=True)
                class_embedding_list.append(embeddings[:, 0, :])
                label_list.append(label)
        class_embedding_list    = torch.cat(class_embedding_list, dim=0) # [L, d]
        label_list              = torch.cat(label_list, dim=0)
        proto_list              = []

        for class_index in range(start_idx, num_seen_cls):
            data_index      = (label_list == class_index).nonzero(as_tuple=False)
            embeddings      = class_embedding_list[data_index.squeeze(-1)] # [N, d]
            embedding_mean  = embeddings.mean(0)
            proto_list.append(embedding_mean)

        proto_list = torch.stack(proto_list, dim=0)
        torch.save({'memory':proto_list.cpu()}, 'memory.pth')

        return proto_list
