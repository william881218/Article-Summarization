import torch
from data import *
from model import *
import random
import time
import math
from torch.nn.utils.rnn import pad_sequence

batch_size = 2
training_data_path = './short.jsonl'
rnn_hidden_dim=32
tag_type_num=2



def pad_collate(batch):
    x_data, y_data = zip(*batch)
    x_data = list(x_data)
    y_data = list(y_data)
    x_data.sort(key=lambda x: len(x), reverse=True)
    y_data.sort(key=lambda x: len(x), reverse=True)
    x_len = [len(x) for x in x_data]
    y_len = [len(y) for y in y_data]
    x_padded = pad_sequence(x_data, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y_data, batch_first=True, padding_value=0)
    return x_padded, y_padded, x_len, y_len

def main():

    #prepare training dataset with DataLoader, which will return a batched, padded sample
    training_dataset = ArticleDataset(training_data_path)
    article_data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

    #prepare embedding layer
    embedding = nn.Embedding(training_dataset.vocab_size, word_vec_d)
    embedding.weight.data.copy_(torch.from_numpy(training_dataset.index2vec.astype('float32')))

    #creating model
    model = RNNTagger(embedding_dim=word_vec_d, hidden_dim=rnn_hidden_dim, output_size=tag_type_num)

    #training start
    for i_ipoch, (x_padded, y_padded, x_lens, y_lens) in enumerate(article_data_loader):
        #Convert word to vector
        x_embed = embedding(x_padded.long())
        x_embed = Variable(x_embed)
        #packing
        x_packed = pack_padded_sequence(x_embed, x_lens, batch_first=True, enforce_sorted=False)

        output = model(x_packed)
        print(output)
        print(output.shape)
        print('ipoch {}'.format(i_ipoch))
        return
    """
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['text'].size(), sample_batched['extractive_gt'].size())
        if i_batch > 3:
            break
    """



if __name__ == '__main__':
    main()


"""
def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor

rnn = RNN(n_letters, n_hidden, n_categories)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()
    optimizer.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.data

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for epoch in range(1, n_epochs + 1):
    category, line, category_tensor, line_tensor = randomTrainingPair()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

torch.save(rnn, 'char-rnn-classification1_6lr=3256.pt')
"""
