import logging
import torch
from torch.utils.data.dataloader import DataLoader
import wandb

import argparse
import json
import spacy
from tqdm import trange
from model import MorioModel
from transformers import AdamW
from sklearn.metrics import classification_report

PATH_TO_DATASET = "/scratch/tc2vg/amazon_argmining_dataset/"

# torch dataset class for easy batching of amazon dataset by graph
class AmazonDataset(torch.utils.data.Dataset):
    def __init__(self, spacy_docs, prop_labels, edges):
        self.docs = spacy_docs # list of tuples of text and spans of dataset
        self.prop_labels = prop_labels # list of prop labels of dataset
        self.edges = edges # list of matricies of edge connections

    def __getitem__(self, idx):
        item ={}
        item['docs'] = self.docs[idx]
        item['prop_labels'] = self.prop_labels[idx]
        item['edges'] = self.edges[idx]
        return item

    def __len__(self):
        return len(self.prop_labels)

# a simple custom collate function to just return a dic
def my_collate(batch):
    '''batch is a list of dictionaries'''
    big_dic = {}

    for key in batch[0].keys():        
        big_dic[key] = [item[key] for item in batch]

    return big_dic

def create_dataloaders():
    data = []
    with open(PATH_TO_DATASET + "processed_train_data.json", 'r') as f:
        data = json.load(f)

    data_test = []
    with open(PATH_TO_DATASET + "processed_test_data.json", 'r') as f:
        data_test = json.load(f)

    # Get labels    
    conversion = {'reference': 0,
                  'fact': 1,
                  'testimony': 2,
                  'value': 3,
                  'policy': 4}

    train_prop_labels = [] # true value
    test_prop_labels = [] # true value

    for example in data:
        train_prop_labels.append([conversion[prop['type']] for prop in example['propositions']])

    for example in data_test:
        test_prop_labels.append([conversion[prop['type']] for prop in example['propositions']])

    train_prop_edges = []
    # train_prop_edge_labels = []

    for example in data:
        num_spans = len(example['propositions'])
        edge_mat = torch.zeros(num_spans, num_spans)
        for i, prop in enumerate(example['propositions']):
            if prop['reasons']:
                for reason in prop['reasons']:
                    edge_mat[int(reason), i] = 1
            
            if prop['evidence']:
                for evidence in prop['evidence']:
                    edge_mat[int(evidence), i] = 1
        train_prop_edges.append(edge_mat)

    test_prop_edges = []
    # test_prop_edge_labels = []

    for example in data_test:
        num_spans = len(example['propositions'])
        edge_mat = torch.zeros(num_spans, num_spans)
        for i, prop in enumerate(example['propositions']):
            if prop['reasons']:
                for reason in prop['reasons']:
                    edge_mat[int(reason), i] = 1
            
            if prop['evidence']:
                for evidence in prop['evidence']:
                    edge_mat[int(evidence), i] = 1
        test_prop_edges.append(edge_mat)

    # get word to idx and pos to idx dicts and spacy doc objects
    nlp = spacy.load("en_core_web_sm")
    
    docs_train = []
    docs_test = []

    for review in data:
        props_doc = []
        for prop in review['propositions']:
            props_doc.append(prop['text'])

        docs_train.append(list(nlp.pipe(props_doc)))

    for review in data_test:
        props_doc = []
        for prop in review['propositions']:
            props_doc.append(prop['text'])
        
        docs_test.append(list(nlp.pipe(props_doc)))


    docs = docs_train + docs_test
    tokens = ['<pad>'] + [token.text for doc in docs for comment in doc for token in comment]
    pos = ['<pad>'] + [token.pos_ for doc in docs for comment in doc for token in comment]

    word_to_idx = {word: idx for idx, word in enumerate(set(tokens))}
    pos_to_idx = {pos: idx for idx, pos in enumerate(set(pos))}

    # set up dataset objects
    train_dataset = AmazonDataset(docs_train, train_prop_labels, train_prop_edges)
    test_dataset = AmazonDataset(docs_test, test_prop_labels, test_prop_edges)

    # set up data loaders

    train_loader = DataLoader(train_dataset, batch_size=16, collate_fn= my_collate, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn= my_collate, shuffle=False)

    return train_loader, test_loader, word_to_idx, pos_to_idx


def train(loader, model, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    
    for i, batch in enumerate(loader, 0):   
        optimizer.zero_grad()         
        
        # missing batch dimenstion, need that for edge predictions
        batch_docs = batch['docs']

        # convert space offset to token offset
        # [docs in batch, comments in doc, (comment offset tuple)]
        token_offsets = []

        for doc in batch_docs:
            doc_token_offsets = []
            offset = 0
            for comment in doc:
                doc_token_offsets.append((offset, offset + len(comment)))
                offset = offset + len(comment)  
            token_offsets.append(doc_token_offsets)  

        s_edge, s_prop, prop_lens = model(batch_docs, token_offsets)

        # calculate proposition type loss
        type_loss = 0
        
        for idx, doc in enumerate(batch['prop_labels']):
            type_loss += criterion(s_prop[idx][:prop_lens[idx]], torch.tensor(doc).to(device))
        
        # calculate edge prediction loss

        # generate masks for not using diagonal
        masks = [torch.ones((length, length)).fill_diagonal_(0).bool().to(device) for length in prop_lens]

        labels = [torch.masked_select(batch_edges.to(device), mask).to(device) for mask, batch_edges in zip(masks, batch['edges'])]
        max_prop_len = max(prop_lens)
        score_masks = [torch.zeros((max_prop_len, max_prop_len)).bool() for length in prop_lens]

        for mask, score_mask in zip(masks, score_masks):
            length = mask.size(0)
            score_mask[:length, :length] = mask
        
        edge_tensors = [s_edge[idx][mask].to(device) for idx, mask in enumerate(score_masks)]
        
        edge_loss = 0
        for edge_tensor, edge_labels in zip(edge_tensors, labels):
            edge_loss += criterion(edge_tensor, edge_labels.long())
        
        loss = type_loss + edge_loss

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss = 0.0
        running_loss += loss.item()

    return running_loss/i

def eval(loader, model, device, print_report=False):
    model.eval()
    edge_preds = []
    edge_labels = []

    prop_preds = []
    prop_labels = []

    with torch.no_grad():
        for j, batch in enumerate(loader):
            # set up batch specific padding structures
            batch_docs = batch['docs']

            # convert space offset to token offset
            
            # [docs in batch, comments in doc, (comment offset tuple)]
            token_offsets = []

            for doc in batch_docs:
                doc_token_offsets = []
                offset = 0
                for comment in doc:
                    doc_token_offsets.append((offset, offset + len(comment)))
                    offset = offset + len(comment)  
                token_offsets.append(doc_token_offsets)  

            s_edge, s_prop, prop_lens = model(batch_docs, token_offsets)

            # calculate prop predictions
            for doc in batch['prop_labels']:
                prop_labels += doc
            
            for idx, length in enumerate(prop_lens):
                prop_preds += torch.argmax(s_prop[idx][:length], dim=1).cpu().tolist()


            # calculate edge predictions
            masks = [torch.ones((length, length)).fill_diagonal_(0).bool().to(device) for length in prop_lens]
            labels = [torch.masked_select(batch_edges.to(device), mask).to(device) for mask, batch_edges in zip(masks, batch['edges'])]
            
            max_prop_len = max(prop_lens)
            score_masks = [torch.zeros((max_prop_len, max_prop_len)).bool() for length in prop_lens]

            for mask, score_mask in zip(masks, score_masks):
                length = mask.size(0)
                score_mask[:length, :length] = mask
            
            edge_tensors = [s_edge[idx][mask].to(device) for idx, mask in enumerate(score_masks)]
            
            for label in labels:
                edge_labels += label.cpu().tolist()

            for edge_tensor in edge_tensors:
                edge_preds += torch.argmax(edge_tensor, dim=1).cpu().tolist()
    
    if print_report:
        print(classification_report(prop_labels, prop_preds))
        logging.info(classification_report(prop_labels, prop_preds))
        print(classification_report(edge_labels, edge_preds))
        logging.info(classification_report(edge_labels, edge_preds))

    prop_report = classification_report(prop_labels, prop_preds, output_dict=True)
    edge_report = classification_report(edge_labels, edge_preds, output_dict=True)

    return prop_report['macro avg']['f1-score'], edge_report['1.0']['f1-score']



def main():
    parser = argparse.ArgumentParser(description='Create and run Morio argument mining model')
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs to train')
    parser.add_argument('--lr', default=12e-4, type=float, help='learning rate for optimizer')
    parser.add_argument('--bert_embedding', action='store_true', help='use bert to encode')
    parser.add_argument('--elmo_embedding', action='store_true', help='use elmo to encode')
    parser.add_argument('--glove_embedding', action='store_true', help='use glove to encode')
    parser.add_argument('--device', '-d', default=0, type=int, help='ID of GPU to use')
    parser.add_argument('--save_dir', default='/scratch/tc2vg/morio-model-amazon-runs/', help='path to saved model files')
    # parser.add_argument('--embed', default='data/glove.6B.100d.txt', help='path to pretrained embeddings')
    # parser.add_argument('--unk', default='unk', help='unk token in pretrained embeddings')
    # parser.add_argument('--bert', default='bert-base-cased', help='which BERT model to use')

    # Get the hyperparameters
    args = parser.parse_args()

    # wandb.init(project="train_argmining_model_amazon", entity="tingtang2")
    # Pass them to wandb.init
    # wandb.init(config=args)
    # Access all hyperparameter values through wandb.config
    # config = wandb.config



    # filename for logging and saved models
    filename = f'morio-model-epochs-{args.epochs}-epochs-{args.lr}-lr'

    if args.bert_embedding:
        filename = 'bert+' + filename 
    if not args.glove_embedding:
        filename = 'noglove-' + filename
    if not args.elmo_embedding:
        filename = 'noelmo-' + filename

    logging.basicConfig(level=logging.DEBUG, filename= './' + filename+'.log', filemode='w', format='%(message)s')
    logging.info(args)
    

    # set up training and model
    device = torch.device(f"cuda:{args.device}")
    train_loader, test_loader, word_to_idx, pos_to_idx = create_dataloaders()

    mm = MorioModel(word_to_idx, pos_to_idx, device=device).to(device)

    print("initialized model")

    # stuff we need for loop
    criterion = mm.criterion
    optimizer = AdamW(mm.parameters(), lr=args.lr)

    for epoch in trange(args.epochs):
        train_loss = train(train_loader, mm, device, optimizer, criterion)
        train_prop_f1, train_edge_f1 = eval(train_loader, mm, device, False)

        test_print_flag = False
        if epoch % 10 == 9:
            test_print_flag = True
        test_prop_f1, test_edge_f1 = eval(test_loader, mm, device, test_print_flag)

        print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f}, Train prop f1: {train_prop_f1:.4f}, Test prop f1: {test_prop_f1:.4f}, Train edge f1: {train_edge_f1:.4f}, Test edge f1: {test_edge_f1:.4f}')
        logging.info(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f}, Train prop f1: {train_prop_f1:.4f}, Test prop f1: {test_prop_f1:.4f}, Train edge f1: {train_edge_f1:.4f}, Test edge f1: {test_edge_f1:.4f}')

    print('Finished Training and Saving Model')
    
    save_path = args.save_dir + filename + '-.pt'
    torch.save(mm.state_dict(), save_path)

if __name__ == '__main__':
    main()