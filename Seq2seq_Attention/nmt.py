#!/u/subramas/miniconda2/bin/python
"""Main script to run things"""


from tqdm import tqdm
from data_utils import read_nmt_data, get_minibatch, read_config
from model import Seq2SeqAttention
from evaluate import evaluate_model
import numpy as np
import logging
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="path to json config",
    required=True
)
args = parser.parse_args()
config_file_path = args.config
config = read_config(config_file_path)
save_dir = config['data']['save_dir']
load_dir = config['data']['load_dir']
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename="/home/ahf/Bug_fix_info/model/save_path/buggy_line/log.txt",
    filemode='w'
)

# define a new Handler to log to console as well
console = logging.StreamHandler()
# optional, set the logging level
console.setLevel(logging.INFO)
# set a format which is the same for console use
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


print('Reading data ...')

src, trg, infotrg = read_nmt_data(
    src=config['data']['src'],
    config=config,
    trg=config['data']['trg'],
    infotrg=config["data"]["infotrg"]
)

src_test, trg_test,infotrg_test = read_nmt_data(
    src=config['data']['test_src'],
    config=config,
    trg=config['data']['test_trg'],
    infotrg=config["data"]["test_infotrg"]
)

batch_size = config['data']['batch_size']
max_length = config['data']['max_src_length']
src_vocab_size = len(src['word2id'])
trg_vocab_size = len(trg['word2id'])
infotrg_vocab_size = len(infotrg["word2id"])
logging.info('Model Parameters : ')
logging.info('Task : %s ' % (config['data']['task']))
logging.info('Model : %s ' % (config['model']['seq2seq']))
logging.info('Source Word Embedding Dim  : %s' % (config['model']['dim_word_src']))
logging.info('Target Word Embedding Dim  : %s' % (config['model']['dim_word_trg']))
logging.info('Source RNN Hidden Dim  : %s' % (config['model']['dim']))
logging.info('Target RNN Hidden Dim  : %s' % (config['model']['dim']))
logging.info('Source RNN Depth : %d ' % (config['model']['n_layers_src']))
logging.info('Target RNN Depth : %d ' % (1))
logging.info('Source RNN Bidirectional  : %s' % (config['model']['bidirectional']))
logging.info('Batch Size : %d ' % (config['model']['n_layers_trg']))
logging.info('Optimizer : %s ' % (config['training']['optimizer']))
logging.info('Learning Rate : %f ' % (config['training']['lrate']))

logging.info('Found %d words in src ' % (src_vocab_size))
logging.info('Found %d words in trg ' % (trg_vocab_size))
logging.info('Found %d words in trg ' % (infotrg_vocab_size))

code_weight_mask = torch.ones(trg_vocab_size).cuda()
info_weight_mask = torch.ones(infotrg_vocab_size).cuda()
code_weight_mask[trg['word2id']['<pad>']] = 0
info_weight_mask[infotrg['word2id']['<pad>']] = 0

code_loss_criterion = nn.CrossEntropyLoss(weight=code_weight_mask).cuda()
info_loss_criterion = nn.CrossEntropyLoss(weight=info_weight_mask).cuda()

model = Seq2SeqAttention(
    src_emb_dim=config['model']['dim_word_src'],
    trg_emb_dim=config['model']['dim_word_trg'],
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    infotrg_vocab_size=infotrg_vocab_size,
    src_hidden_dim=config['model']['dim'],
    trg_hidden_dim=config['model']['dim'],
    ctx_hidden_dim=config['model']['dim'],
    attention_mode='dot',
    batch_size=batch_size,
    bidirectional=config['model']['bidirectional'],
    pad_token_src=src['word2id']['<pad>'],
    pad_token_trg=trg['word2id']['<pad>'],
    pad_token_infoarg=infotrg['word2id']['<pad>'],
    nlayers=config['model']['n_layers_src'],
    dropout=0.,
).cuda()


if load_dir:
    model.load_state_dict(torch.load(
        open(load_dir)
    ))

# __TODO__ Make this more flexible for other learning methods.
if config['training']['optimizer'] == 'adam':
    lr = config['training']['lrate']
    optimizer = optim.Adam(model.parameters(), lr=lr)
elif config['training']['optimizer'] == 'adadelta':
    optimizer = optim.Adadelta(model.parameters())
elif config['training']['optimizer'] == 'sgd':
    lr = config['training']['lrate']
    optimizer = optim.SGD(model.parameters(), lr=lr)
else:
    raise NotImplementedError("Learning method not recommend for task")
max_code_bleu,patience,best_model_config,best_model_filepath = 0,3,"","/home/ahf/Bug_fix_info/model/save_path/buggy_line/model_saveconfig.txt"
best_model_fp = open(best_model_filepath,"w")
for i in tqdm(range(1000)):
    losses = []
    for j in range(0, len(src['data']), batch_size):

        input_lines_src, _, lens_src, mask_src = get_minibatch(
            src['data'], src['word2id'], j,
            batch_size, max_length, add_start=True, add_end=True
        )
        input_lines_trg, output_lines_trg, lens_trg, mask_trg = get_minibatch(
            trg['data'], trg['word2id'], j,
            batch_size, max_length, add_start=True, add_end=True
        )
        input_lines_infotrg, output_lines_infotrg, lens_infotrg, mask_infotrg = get_minibatch(
            infotrg['data'], infotrg['word2id'], j,
            batch_size, max_length, add_start=True, add_end=True
        )
        code_decoder_logit,info_decoder_logit = model(input_lines_src, input_lines_trg,input_lines_infotrg)
        optimizer.zero_grad()
        loss = code_loss_criterion(
            code_decoder_logit.contiguous().view(-1, trg_vocab_size),
            output_lines_trg.view(-1)
        )+info_loss_criterion(
            info_decoder_logit.contiguous().view(-1, infotrg_vocab_size),
            output_lines_infotrg.view(-1)
        )
        losses.append(loss.data[0])
        loss.backward()
        optimizer.step()

        if j % config['management']['monitor_loss'] == 0:
            logging.info('Epoch : %d Minibatch : %d Loss : %.5f' % (
                i, j, np.mean(losses))
            )
            losses = []

        if (
            config['management']['print_samples'] and
            j % config['management']['print_samples'] == 0
        ):
            code_probs = model.code_decode(
                code_decoder_logit
            ).data.cpu().numpy().argmax(axis=-1)
            info_probs = model.info_decode(
                info_decoder_logit
            ).data.cpu().numpy().argmax(axis=-1)

            output_lines_trg = output_lines_trg.data.cpu().numpy()
            output_lines_infotrg = output_lines_infotrg.data.cpu().numpy()
            for sentence_pred, sentence_real in zip(
                code_probs[:5], output_lines_trg[:5]
            ):
                sentence_pred = [trg['id2word'][x] for x in sentence_pred]
                sentence_real = [trg['id2word'][x] for x in sentence_real]

                if '</s>' in sentence_real:
                    index = sentence_real.index('</s>')
                    sentence_real = sentence_real[:index]
                    sentence_pred = sentence_pred[:index]

                logging.info('Predicted code: %s ' % (' '.join(sentence_pred)))
                logging.info('-----------------------------------------------')
                logging.info('Real code: %s ' % (' '.join(sentence_real)))
                logging.info('===============================================')
            for sentence_pred, sentence_real in zip(
                info_probs[:5], output_lines_infotrg[:5]
            ):
                sentence_pred = [infotrg['id2word'][x] for x in sentence_pred]
                sentence_real = [infotrg['id2word'][x] for x in sentence_real]

                if '</s>' in sentence_real:
                    index = sentence_real.index('</s>')
                    sentence_real = sentence_real[:index]
                    sentence_pred = sentence_pred[:index]

                logging.info('Predicted buginfo: %s ' % (' '.join(sentence_pred)))
                logging.info('-----------------------------------------------')
                logging.info('Real buginfo: %s ' % (' '.join(sentence_real)))
                logging.info('===============================================')
        if j % config['management']['checkpoint_freq'] == 0:

            logging.info('Evaluating model ...')
            code_bleu,info_bleu = evaluate_model(
                model, src, src_test, trg,infotrg,
                trg_test,infotrg_test, config, verbose=False,
                metric='bleu',
            )

            logging.info('Epoch : %d Minibatch : %d : BLEU : %.5f ' % (i, j, code_bleu))
            logging.info('Epoch : %d Minibatch : %d : BLEU : %.5f ' % (i, j, info_bleu))
            logging.info('Saving model ...')

            torch.save(
                model.state_dict(),
                open(os.path.join(
                    save_dir,
                    '__epoch_%d__minibatch_%d' % (i, j) + '.model'), 'wb'
                )
            )

    code_bleu,info_bleu = evaluate_model(
        model, src, src_test, trg,infotrg,
        trg_test,infotrg_test, config, verbose=False,
        metric='bleu',
    )
    """
    early stop
    """
    if code_bleu >= max_code_bleu:
        max_code_bleu = code_bleu
        patience = 3
        best_model_config = os.path.join(save_dir,'__epoch_%d' % (i) + '.model')
        best_model_fp.write(best_model_config+"\n")
    else:
        patience -= 1
    if patience == 0:
        break
    logging.info('Epoch : %d : BLEU : %.5f ' % (i, code_bleu))
    logging.info('Epoch : %d : BLEU : %.5f ' % (i, info_bleu))

    torch.save(
        model.state_dict(),
        open(os.path.join(
            save_dir,'__epoch_%d' % (i) + '.model'), 'wb'
        )
    )
best_model_fp.close()