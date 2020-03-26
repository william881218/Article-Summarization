from model import *
from data import *
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import sys

PREDICT_DATASET_PATH = './valid_short.jsonl'
OUTPUT_PATH = './output.jsonl'
MODEL_PATH = './best.pt'
batch_size = 2

def pad_collate_predicting(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    x_data, idx = zip(*batch)
    x_len = [len(x) for x in x_data]
    x_padded = pad_sequence(x_data, batch_first=True, padding_value=0)
    return x_padded, x_len, idx

if __name__ == '__main__':
    #create predicting dataset and dataloader
    predicting_dataset = ArticleDataset(PREDICT_DATASET_PATH, predicting=True)
    article_data_loader = DataLoader(predicting_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_predicting)

    #prepare embedding model and predicting model
    embedding = WordEmbedding(predicting_dataset.vocab_size, word_vec_d, predicting_dataset.index2vec)
    rnn = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

    #if gpu is available
    if torch.cuda.is_available():
        rnn = rnn.cuda()
        embedding = embedding.cuda()

    for i_batch, (x_padded, x_len, idx) in enumerate(article_data_loader):
        with torch.no_grad():
            x_padded = x_padded.long()
            if torch.cuda.is_available():
                x_padded = x_padded.cuda()
            #Convert word to vector
            x_embed = embedding(x_padded)

            #packing
            x_packed = pack_padded_sequence(x_embed, x_len, batch_first=True, enforce_sorted=False)

            #Feeding into rnn and get an output
            output, output_lengths = rnn(x_packed)

            for i_sent, sent_len in enumerate(output_lengths):
                predicted_gt = predicting_dataset.predict(idx[i_sent], output[i_sent, 0:sent_len])
                print('id: {}, prediction: {}'.format(predicting_dataset.article(idx[i_sent]).id, predicted_gt))
