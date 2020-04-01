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
    x_padded = pad_sequence(x_data, batch_first=True, padding_value=PAD_ID)
    return x_padded, x_len, idx

def evaluate(seq2seq, batch):
    with torch.no_grad():
        x_padded, x_len, idx = batch
        if torch.cuda.is_available():
            x_padded = x_padded.cuda(GPU_DEVICE)

        decoder_output, hidden = seq2seq.predict((x_padded, x_len))

        topv, topi = torch.topk(decoder_output, 1)
        outputs = topi.squeeze().transpose(0, 1).contiguous()

        return outputs, idx

def main(model_path=MODEL_PATH, input_file_path=INPUT_FILE_PATH, output_path=OUTPUT_PATH):

    #create predicting dataset and dataloader
    predicting_dataset = ArticleDataset(input_file_path, predicting=True)
    predicting_data_loader = DataLoader(predicting_dataset, batch_size=predict_batch_size,
                                            shuffle=False, collate_fn=pad_collate_predicting)

    #load the model
    seq2seq = torch.load(model_path, map_location=torch.device('cpu'))

    #if gpu is available
    if torch.cuda.is_available():
        seq2seq = seq2seq.cuda(GPU_DEVICE)

    output_list = []

    for i_batch, batch in enumerate(predicting_data_loader):

        #no need to backprop, so no_grad
        outputs, idx = evaluate(seq2seq, batch)

        for i, output in enumerate(outputs):
            output_str = []
            for word in output:
                if word.item() == EOS_ID:
                    break
                output_str += [seq2seq.index2word[word.item()]]
            output_list += ['\"id\": \"{}\", \"predict\": \"{}\"'.format(idx[i], ' '.join(output_str))]

    with open(output_path, 'w') as output_file:
        for output in output_list:
            output_file.write('{' + output + '}\n')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print('usage: python predict.py [model path] [input file path] [output file path]')
        print('if path not assigned, model path: {}, input file path: {}, output file path: {}'.format(MODEL_PATH, INPUT_FILE_PATH, OUTPUT_PATH))
        exit()
