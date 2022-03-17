from linecache import cache
from unicodedata import bidirectional
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence, pad_sequence
from torch.nn.modules.rnn import apply_permutation
from torch.nn import functional as F
import torchtext
from transformers import BertTokenizer, BertModel
from transformers.models import bert



class ScalarMix(nn.Module):
    r"""
    Computes a parameterized scalar mixture of :math:`N` tensors, :math:`mixture = \gamma * \sum_{k}(s_k * tensor_k)`
    where :math:`s = \mathrm{softmax}(w)`, with :math:`w` and :math:`\gamma` scalar parameters.
    Args:
        n_layers (int):
            The number of layers to be mixed, i.e., :math:`N`.
        dropout (float):
            The dropout ratio of the layer weights.
            If dropout > 0, then for each scalar weight, adjusts its softmax weight mass to 0
            with the dropout probability (i.e., setting the unnormalized weight to -inf).
            This effectively redistributes the dropped probability mass to all other weights.
            Default: 0.
    """

    def __init__(self, n_layers, dropout=0):
        super().__init__()

        self.n_layers = n_layers

        self.weights = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.tensor([1.0]))
        self.dropout = nn.Dropout(dropout)

    def __repr__(self):
        s = f"n_layers={self.n_layers}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"

        return f"{self.__class__.__name__}({s})"

    def forward(self, tensors):
        r"""
        Args:
            tensors (list[~torch.Tensor]):
                :math:`N` tensors to be mixed.
        Returns:
            The mixture of :math:`N` tensors.
        """

        normed_weights = self.dropout(self.weights.softmax(-1))
        weighted_sum = sum(w * h for w, h in zip(normed_weights, tensors))

        return self.gamma * weighted_sum

class MLP(nn.Module):
    r"""
    Applies a linear transformation together with a non-linear activation to the incoming tensor:
    :math:`y = \mathrm{Activation}(x A^T + b)`
    Args:
        n_in (~torch.Tensor):
            The size of each input feature.
        n_out (~torch.Tensor):
            The size of each output feature.
        dropout (float):
            If non-zero, introduces a :class:`SharedDropout` layer on the output with this dropout ratio. Default: 0.
        activation (bool):
            Whether to use activations. Default: True.
    """

    def __init__(self, n_in, n_out, dropout=0, activation=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.LeakyReLU(negative_slope=0.1) if activation else nn.Identity()
        self.dropout = SharedDropout(p=dropout)

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        r"""
        Args:
            x (~torch.Tensor):
                The size of each input feature is `n_in`.
        Returns:
            A tensor with the size of each output feature `n_out`.
        """

        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x

# Biaffine module from SuPar
class Biaffine(nn.Module):
    r"""
    Biaffine layer for first-order scoring.
    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y)` of the vector pair :math:`(x, y)` is computed as :math:`x^T W y`,
    in which :math:`x` and :math:`y` can be concatenated with bias terms.
    References:
        - Timothy Dozat and Christopher D. Manning. 2017.
          `Deep Biaffine Attention for Neural Dependency Parsing`_.
    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The number of output channels.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``True``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``True``.
    .. _Deep Biaffine Attention for Neural Dependency Parsing:
        https://openreview.net/forum?id=Hk95PK9le
    """

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=False):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in+bias_x, n_in+bias_y))

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.
        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s

# Just TSP 
class TSP(nn.Module):
    def __init__(self, input_size, lin_size, device):
        super(TSP, self).__init__()
        
        self.start = nn.Linear(input_size, lin_size)
        self.v = nn.Linear(lin_size, 1, bias = False)
        self.device = device
        

    def forward(self, word_reps, token_offsets):
        alpha = self.v(self.start(word_reps))
        alpha = alpha.reshape(alpha.size()[:-1])

        self.s = torch.empty(alpha.size()).to(self.device)

        for idx, doc in enumerate(token_offsets):
            for offset in doc:
                self.s[idx][offset[0]:offset[1]] = F.softmax(alpha[idx][offset[0]:offset[1]], dim=0)

        prop_lens = [len(doc) for doc in token_offsets]
        
        
        self.span_rep = torch.zeros(len(token_offsets), max(prop_lens), word_reps.size(-1)).to(self.device)

        for i, doc in enumerate(token_offsets):
            for j, offset in enumerate(doc):
                self.span_rep[i][j] = torch.sum(word_reps[i][offset[0]:offset[1]] * self.s[i][offset[0]:offset[1], None], dim=0)

        offset_lens = [torch.tensor([offset[1] - offset[0] for offset in doc]).to(self.device) for doc in token_offsets]

        phi = pad_sequence(offset_lens, True).to(self.device)
        phi = phi.reshape(phi.size(0), phi.size(1), 1)

        span_ends = [torch.stack([word_reps[idx][offset[1]-1] for offset in doc]).to(self.device) for idx, doc in enumerate(token_offsets)]

        span_ends_padded = pad_sequence(span_ends, True)

        final_span_rep = torch.cat((span_ends_padded, self.span_rep, phi), dim=-1)
                
        return final_span_rep, prop_lens

class SharedDropout(nn.Module):
    r"""
    SharedDropout differs from the vanilla dropout strategy in that the dropout mask is shared across one dimension.
    Args:
        p (float):
            The probability of an element to be zeroed. Default: 0.5.
        batch_first (bool):
            If ``True``, the input and output tensors are provided as ``[batch_size, seq_len, *]``.
            Default: ``True``.
    Examples:
        >>> x = torch.ones(1, 3, 5)
        >>> nn.Dropout()(x)
        tensor([[[0., 2., 2., 0., 0.],
                 [2., 2., 0., 2., 2.],
                 [2., 2., 2., 2., 0.]]])
        >>> SharedDropout()(x)
        tensor([[[2., 0., 2., 0., 2.],
                 [2., 0., 2., 0., 2.],
                 [2., 0., 2., 0., 2.]]])
    """

    def __init__(self, p=0.5, batch_first=True):
        super().__init__()

        self.p = p
        self.batch_first = batch_first

    def __repr__(self):
        s = f"p={self.p}"
        if self.batch_first:
            s += f", batch_first={self.batch_first}"

        return f"{self.__class__.__name__}({s})"

    def forward(self, x):
        r"""
        Args:
            x (~torch.Tensor):
                A tensor of any shape.
        Returns:
            The returned tensor is of the same shape as `x`.
        """

        if self.training:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p).unsqueeze(1)
            else:
                mask = self.get_mask(x[0], self.p)
            x = x * mask

        return x

    @staticmethod
    def get_mask(x, p):
        return x.new_empty(x.shape).bernoulli_(1 - p) / (1 - p)



class VariationalLSTM(nn.Module):
    r"""
    VariationalLSTM :cite:`yarin-etal-2016-dropout` is an variant of the vanilla bidirectional LSTM
    adopted by Biaffine Parser with the only difference of the dropout strategy.
    It drops nodes in the LSTM layers (input and recurrent connections)
    and applies the same dropout mask at every recurrent timesteps.
    APIs are roughly the same as :class:`~torch.nn.LSTM` except that we only allows
    :class:`~torch.nn.utils.rnn.PackedSequence` as input.
    Args:
        input_size (int):
            The number of expected features in the input.
        hidden_size (int):
            The number of features in the hidden state `h`.
        num_layers (int):
            The number of recurrent layers. Default: 1.
        bidirectional (bool):
            If ``True``, becomes a bidirectional LSTM. Default: ``False``
        dropout (float):
            If non-zero, introduces a :class:`SharedDropout` layer on the outputs of each LSTM layer except the last layer.
            Default: 0.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, dropout=0):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.num_directions = 1 + self.bidirectional

        self.f_cells = nn.ModuleList()
        if bidirectional:
            self.b_cells = nn.ModuleList()
        for _ in range(self.num_layers):
            self.f_cells.append(nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))
            if bidirectional:
                self.b_cells.append(nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))
            input_size = hidden_size * self.num_directions

        self.reset_parameters()

    def __repr__(self):
        s = f"{self.input_size}, {self.hidden_size}"
        if self.num_layers > 1:
            s += f", num_layers={self.num_layers}"
        if self.bidirectional:
            s += f", bidirectional={self.bidirectional}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        for param in self.parameters():
            # apply orthogonal_ to weight
            if len(param.shape) > 1:
                nn.init.orthogonal_(param)
            # apply zeros_ to bias
            else:
                nn.init.zeros_(param)

    def permute_hidden(self, hx, permutation):
        if permutation is None:
            return hx
        h = apply_permutation(hx[0], permutation)
        c = apply_permutation(hx[1], permutation)

        return h, c

    def layer_forward(self, x, hx, cell, batch_sizes, reverse=False):
        hx_0 = hx_i = hx
        hx_n, output = [], []
        steps = reversed(range(len(x))) if reverse else range(len(x))
        if self.training:
            hid_mask = SharedDropout.get_mask(hx_0[0], self.dropout)

        for t in steps:
            last_batch_size, batch_size = len(hx_i[0]), batch_sizes[t]
            if last_batch_size < batch_size:
                hx_i = [torch.cat((h, ih[last_batch_size:batch_size])) for h, ih in zip(hx_i, hx_0)]
            else:
                hx_n.append([h[batch_size:] for h in hx_i])
                hx_i = [h[:batch_size] for h in hx_i]
            
            hx_i = [h for h in cell(x[t], hx_i)]
            output.append(hx_i[0])
            if self.training:
                hx_i[0] = hx_i[0] * hid_mask[:batch_size]
        if reverse:
            hx_n = hx_i
            output.reverse()
        else:
            hx_n.append(hx_i)
            hx_n = [torch.cat(h) for h in zip(*reversed(hx_n))]
        output = torch.cat(output)

        return output, hx_n

    def forward(self, sequence, hx=None):
        r"""
        Args:
            sequence (~torch.nn.utils.rnn.PackedSequence):
                A packed variable length sequence.
            hx (~torch.Tensor, ~torch.Tensor):
                A tuple composed of two tensors `h` and `c`.
                `h` of shape ``[num_layers*num_directions, batch_size, hidden_size]`` holds the initial hidden state
                for each element in the batch.
                `c` of shape ``[num_layers*num_directions, batch_size, hidden_size]`` holds the initial cell state
                for each element in the batch.
                If `hx` is not provided, both `h` and `c` default to zero.
                Default: ``None``.
        Returns:
            ~torch.nn.utils.rnn.PackedSequence, (~torch.Tensor, ~torch.Tensor):
                The first is a packed variable length sequence.
                The second is a tuple of tensors `h` and `c`.
                `h` of shape ``[num_layers*num_directions, batch_size, hidden_size]`` holds the hidden state for `t=seq_len`.
                Like output, the layers can be separated using ``h.view(num_layers, num_directions, batch_size, hidden_size)``
                and similarly for c.
                `c` of shape ``[num_layers*num_directions, batch_size, hidden_size]`` holds the cell state for `t=seq_len`.
        """
        x, batch_sizes = sequence.data, sequence.batch_sizes.tolist()
        batch_size = batch_sizes[0]
        h_n, c_n = [], []

        if hx is None:
            ih = x.new_zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
            h, c = ih, ih
        else:
            h, c = self.permute_hidden(hx, sequence.sorted_indices)
        h = h.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        c = c.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)

        for i in range(self.num_layers):
            x = torch.split(x, batch_sizes)
            if self.training:
                mask = SharedDropout.get_mask(x[0], self.dropout)
                x = [i * mask[:len(i)] for i in x]
            x_i, (h_i, c_i) = self.layer_forward(x, (h[i, 0], c[i, 0]), self.f_cells[i], batch_sizes)
            if self.bidirectional:
                x_b, (h_b, c_b) = self.layer_forward(x, (h[i, 1], c[i, 1]), self.b_cells[i], batch_sizes, True)
                x_i = torch.cat((x_i, x_b), -1)
                h_i = torch.stack((h_i, h_b))
                c_i = torch.stack((c_i, c_b))
            x = x_i
            h_n.append(h_i)
            c_n.append(c_i)

        x = PackedSequence(x, sequence.batch_sizes, sequence.sorted_indices, sequence.unsorted_indices)
        hx = torch.cat(h_n, 0), torch.cat(c_n, 0)
        hx = self.permute_hidden(hx, sequence.unsorted_indices)

        return x, hx


class ELMoEmbedding(nn.Module):
    r"""
    Contextual word embeddings using word-level bidirectional LM :cite:`peters-etal-2018-deep`.
    Args:
        model (str):
            The name of the pretrained ELMo registered in `OPTION` and `WEIGHT`. Default: ``'original_5b'``.
        bos_eos (tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of sentence outputs.
            Default: ``(True, True)``.
        n_out (int):
            The requested size of the embeddings. If 0, uses the default size of ELMo outputs. Default: 0.
        dropout (float):
            The dropout to be applied to the ELMo representations. Default: 0.5.
        requires_grad (bool):
            If ``True``, the model parameters will be updated together with the downstream task. Default: ``False``.
    """

    OPTION = {
        'small': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json',  # noqa
        'medium': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json',  # noqa
        'original': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json',  # noqa
        'original_5b': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json',  # noqa
    }
    WEIGHT = {
        'small': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5',  # noqa
        'medium': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5',  # noqa
        'original': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',  # noqa
        'original_5b': 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5',  # noqa
    }

    def __init__(self, device, model='original', dropout=0.45, requires_grad=False, num_output_representations=3):
        super().__init__()

        from allennlp.modules.elmo import Elmo, batch_to_ids

        self.elmo = Elmo(options_file=self.OPTION[model],
                         weight_file=self.WEIGHT[model],
                         num_output_representations=num_output_representations,
                         dropout=dropout,
                         requires_grad=requires_grad,
                         keep_sentence_boundaries=False)

        self.model = model
        self.hidden_size = self.elmo.get_output_dim()
        self.dropout = dropout
        self.requires_grad = requires_grad
        self.batch_to_ids = batch_to_ids
        self.size = 1024
        self.device = device

        self.scalar_mix = ScalarMix(num_output_representations)

    def forward(self, tokenized_spans):
        r"""
        Args:
            chars (~torch.Tensor): ``[batch_size, seq_len, fix_len]``.
        Returns:
            ~torch.Tensor:
                ELMo embeddings of shape ``[batch_size, seq_len, n_out]``.
        """

        batch_elmo_ids = [self.batch_to_ids(tokenized_doc).to(self.device) for tokenized_doc in tokenized_spans]
        batch_elmo_embeddings = [self.elmo(doc_ids) for doc_ids in batch_elmo_ids]
        batch_elmo_embeddings_mixed = [self.scalar_mix([layer[elmo_embedding['mask']] for layer in elmo_embedding['elmo_representations']]) for elmo_embedding in batch_elmo_embeddings]
        
        return batch_elmo_embeddings_mixed


class MorioModel(nn.Module):
    def __init__(self,
                word_to_idx,
                pos_to_idx,
                device,
                bert_embedding = False,
                elmo_embedding = True,
                glove_embedding = True,
                surface_embedding_size = 100,
                pos_embedding_size = 100,
                glove_size = 300,
                lstm_hidden_size = 400,
                lstm_num_layers = 1,
                lstm_dropout = 0.33,
                lstm_output_dropout = 0.25,
                lstm_type_hidden_size = 300,
                lstm_type_num_layers = 3,
                linear_tsp_size = 100,
                mlp_num_layers = 700,
                mlp_dropout = 0.45
                ):
        super(MorioModel, self).__init__()

        self.device = device

        # bert stuff
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_embedding = bert_embedding
        
        # include elmo/glove or not
        self.elmo_embedding = elmo_embedding
        self.glove_embedding = glove_embedding

        bert_size = 0
        if bert_embedding:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.bert = self.bert.requires_grad_(True)
            bert_size = 768


        # preprocessing stuff
        self.word_to_idx = word_to_idx
        self.pos_to_idx = pos_to_idx

        # embeddings
        self.surface_embeddings = nn.Embedding(len(word_to_idx), surface_embedding_size)
        self.surface_mlp = MLP(surface_embedding_size, mlp_num_layers)
        
        self.pos_embeddings = nn.Embedding(len(pos_to_idx), pos_embedding_size)
        self.pos_mlp = MLP(pos_embedding_size, mlp_num_layers)

        self.glove = torchtext.vocab.GloVe(name='840B', dim=glove_size, cache='/scratch/tc2vg/.vector_cache/')
        self.elmo = ELMoEmbedding(device=device)

        # encoder
        input_size = mlp_num_layers + mlp_num_layers 
        if glove_embedding:
            input_size += glove_size

        if elmo_embedding: 
            input_size += self.elmo.size 
        if bert_embedding:
            input_size += bert_size
        
        self.encoder = VariationalLSTM(input_size, lstm_hidden_size, lstm_num_layers, True, lstm_dropout)
        self.lstm_output_dropout = SharedDropout(p=lstm_output_dropout)
        
        # TSP stuff
        self.TSP_prop = TSP(2*lstm_hidden_size, linear_tsp_size, device)
        self.TSP_edge = TSP(2*lstm_hidden_size, linear_tsp_size, device)

        tsp_out = 2 * (2 * lstm_hidden_size) + 1
        self.edge_lstm = VariationalLSTM(tsp_out, lstm_type_hidden_size, lstm_type_num_layers, True, lstm_dropout)
        self.prop_lstm = VariationalLSTM(tsp_out, lstm_type_hidden_size, lstm_type_num_layers, True, lstm_dropout)

        # PLBA
        self.edge_mlp_d = MLP(n_in=2*lstm_type_hidden_size, n_out=mlp_num_layers, dropout=mlp_dropout, activation=True)
        self.edge_mlp_h = MLP(n_in=2*lstm_type_hidden_size, n_out=mlp_num_layers, dropout=mlp_dropout, activation=True)
        self.edge_attn = Biaffine(n_in=mlp_num_layers, n_out=2, bias_x=True, bias_y=True)

        # prop type prediction
        self.prop_mlp = MLP(n_in=2*lstm_type_hidden_size, n_out=mlp_num_layers, dropout=mlp_dropout, activation=True)
        self.prop_lin_proj = MLP(n_in=mlp_num_layers, n_out=5, dropout=0.0, activation=False)
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, docs, token_offset):
        preprocessed_batch, batch_seq_lens = self.preprocess_batch(docs)

        packed_batch = pack_padded_sequence(preprocessed_batch, batch_seq_lens, batch_first=True, enforce_sorted=False).cuda()

        encoded_batch, _ = self.encoder(packed_batch)
        padded_batch, _ = pad_packed_sequence(encoded_batch, batch_first=True)
        padded_batch = self.lstm_output_dropout(padded_batch)

        # TSP 
        prop_tsp, batch_prop_lens = self.TSP_prop(padded_batch, token_offset)
        edge_tsp, _ = self.TSP_edge(padded_batch, token_offset)

        packed_prop_tsp = pack_padded_sequence(prop_tsp, batch_prop_lens, batch_first=True, enforce_sorted=False)
        packed_edge_tsp = pack_padded_sequence(edge_tsp, batch_prop_lens, batch_first=True, enforce_sorted=False)

        encoded_prop_tsp, _ = self.prop_lstm(packed_prop_tsp)
        encoded_edge_tsp, _ = self.edge_lstm(packed_edge_tsp)

        padded_prop_tsp, _ = pad_packed_sequence(encoded_prop_tsp, batch_first=True)
        padded_edge_tsp, _ = pad_packed_sequence(encoded_edge_tsp, batch_first=True)

        edge_d = self.edge_mlp_d(padded_edge_tsp)
        edge_h = self.edge_mlp_h(padded_edge_tsp)
        
        # [batch_size, seq_len, seq_len, 2]
        s_edge = self.edge_attn(edge_d, edge_h).permute(0, 2, 3, 1)
        
        s_prop = self.prop_mlp(padded_prop_tsp)
        s_prop = self.prop_lin_proj(s_prop)
        
        return s_edge, s_prop, batch_prop_lens


    def preprocess_batch(self, docs):
        # [docs in batch x tokens in doc]
        batch_tokens = [[token.text for comment in doc for token in comment] for doc in docs]
        batch_pos = [[token.pos_ for comment in doc for token in comment] for doc in docs]
        
        # token lengths of each document in batch 
        batch_doc_lens = [len(doc) for doc in batch_tokens]

        word_idx = [torch.tensor([self.word_to_idx[token] if token in list(self.word_to_idx.keys()) else 0 for token in comment], dtype=torch.long).to(self.device) for comment in batch_tokens]  
        pos_idx = [torch.tensor([self.pos_to_idx[pos] for pos in comment], dtype=torch.long).to(self.device) for comment in batch_pos]  
        
        # surface and POS embeddings
        batch_surface_embeddings = [self.surface_mlp(self.surface_embeddings(idx)) for idx in word_idx]
        batch_pos_embeddings = [self.pos_mlp(self.pos_embeddings(idx)) for idx in pos_idx]
        
        batch_surface_embeddings_padded = pad_sequence(batch_surface_embeddings, batch_first=True)
        batch_pos_embeddings_padded = pad_sequence(batch_pos_embeddings, batch_first=True)

        final_rep = torch.cat((batch_surface_embeddings_padded, batch_pos_embeddings_padded), dim=-1)
        
        # elmo embeddings
        if self.elmo_embedding:
            tokenized_props = [[[token.text for token in prop] for prop in doc] for doc in docs]

            batch_elmo_embeddings = self.elmo(tokenized_props)
            batch_elmo_embeddings_padded = pad_sequence(batch_elmo_embeddings, batch_first=True)
            final_rep = torch.cat((final_rep, batch_elmo_embeddings_padded), dim=-1)

        # glove embeddings
        if self.glove_embedding:
            glove_reps = [self.glove.get_vecs_by_tokens(doc, lower_case_backup=True).to(self.device) for doc in batch_tokens]
            glove_padded_reps = pad_sequence(glove_reps, batch_first=True)
            
            final_rep = torch.cat((final_rep, glove_padded_reps), dim=-1)

        # bert embeddings 
        if self.bert_embedding:
            batch_bert_tokenized = [self.bert_tokenizer(doc, truncation=True, padding=True, return_tensors="pt").to(self.device) for doc in batch_tokens]
            output = [self.bert(**input_)[1] for input_ in batch_bert_tokenized]
            batch_bert_embeddings = pad_sequence(output, batch_first=True)
            final_rep = torch.cat((final_rep, batch_bert_embeddings), dim=-1)

        # [batch length, max num of words in prop span, overall embedding size]
        return final_rep, batch_doc_lens
    

