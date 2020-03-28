from model import *
from data import *
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import sys
import json

INPUT_FILE_PATH = './test_short.jsonl'
OUTPUT_PATH = './output.jsonl'
MODEL_PATH = './best.pt'
predict_batch_size = 2

def pad_collate_predicting(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    x_data, idx = zip(*batch)
    x_len = [len(x) for x in x_data]
    x_padded = pad_sequence(x_data, batch_first=True, padding_value=0)
    return x_padded, x_len, idx

def main(model_path=MODEL_PATH, input_file_path=INPUT_FILE_PATH, output_path=OUTPUT_PATH):

    #create predicting dataset and dataloader
    predicting_dataset = ArticleDataset(input_file_path, predicting=True)
    article_data_loader = DataLoader(predicting_dataset, batch_size=predict_batch_size, shuffle=False, collate_fn=pad_collate_predicting)

    #prepare embedding model and predicting model
    embedding = WordEmbedding(predicting_dataset.vocab_size, word_vec_d, predicting_dataset.index2vec)
    rnn = torch.load(model_path, map_location=torch.device('cpu'))

    #if gpu is available
    if torch.cuda.is_available():
        rnn = rnn.cuda()
        embedding = embedding.cuda()

    output_list = []
    for i_batch, (x_padded, x_len, idx) in enumerate(article_data_loader):

        #no need to backprop, so no_grad
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
                output_list.append('{' + '"id": "{}", "predict_sentence_index": [{}]'.format(predicting_dataset.article(idx[i_sent]).id, predicted_gt) + '}\n')
                print({'{}_th prediction completed.'.format(i_sent)})

    with open(output_path, 'w') as output_file:
        for output in output_list:
            output_file.write(output)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print('usage: python predict.py [model path] [input file path] [output file path]')
        print('if path not assigned, model path: {}, input file path: {}, output file path: {}'.format(MODEL_PATH, INPUT_FILE_PATH, OUTPUT_PATH))
        exit()
