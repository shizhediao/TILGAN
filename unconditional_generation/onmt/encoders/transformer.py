"""
Implementation of "Attention is All You Need"
"""
import torch
import torch.nn as nn

#from lang.onmt.encoders.encoder import EncoderBase
# from lang.onmt.modules.multi_headed_attn import MultiHeadedAttention
# from lang.onmt.modules.position_ffn import PositionwiseFeedForward
# from lang.onmt.utils.misc import sequence_mask
from onmt.encoders.encoder import EncoderBase
from onmt.modules.multi_headed_attn import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 max_relative_positions=0):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, attn_type="self")
        out = self.dropout(context) + inputs
        return self.feed_forward(out)

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, add_noise, num_layers, d_model, heads, d_ff, dropout,
                 attention_dropout, embeddings, max_relative_positions, aehidden):
        super(TransformerEncoder, self).__init__()
        self.aehidden = aehidden
        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout, attention_dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        if add_noise==True:
            self.squeeze_hidden = nn.Linear(d_model, aehidden*2)
        elif add_noise==False:
            self.squeeze_hidden = nn.Linear(d_model, aehidden)

        self.activation = nn.Tanh()
        # self.unsqueeze_hidden = nn.Linear(aehidden, d_model)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions)

    def forward(self, src, add_noise, soft, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)
        emb = self.embeddings(src, soft=soft)  #emb [16,64,512] = [max_len, batchsize, d_emb]
        # max_len = lengths.max()  #added by shizhe
        max_len = src.shape[0] ## added by shizhe
        batch_size = src.shape[1]
        # print(max_len)
        out = emb.transpose(0, 1).contiguous()  #out [64,33, 512]
        # if(max_len!=16):
        #     print(max_len)
        mask = ~sequence_mask(lengths, max_len).unsqueeze(1)
        # mask = ~sequence_mask(lengths).unsqueeze(1) #(64,1,33)
        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            out = layer(out, mask)

        if add_noise==True:
            out = self.layer_norm(out)   #out [64, 33, 512]
            memory_bank_ori = out.transpose(0, 1).contiguous()  # [33,64, 512]
            memory_bank_ori = self.squeeze_hidden(memory_bank_ori)  #The original encoder output [33,64,200]
            memory_bank_ori = self.activation(memory_bank_ori)
            #@shizhe add noise
            noise = torch.ones(max_len, batch_size, self.aehidden).normal_(0, 1).cuda()  #[33, 64, 100]
            mean = memory_bank_ori.view(max_len, batch_size, 2, self.aehidden)[:, :, 0]  #[33,64,100]
            var = memory_bank_ori.view(max_len, batch_size, 2, self.aehidden)[:, :, 1]  #[33,64,100]
            memory_bank = mean + var * noise
            # memory_bank = self.activation(self.squeeze_hidden(memory_bank))  #[16,64,300]
            # memory_bank = self.unsqueeze_hidden(memory_bank)  # [16,64,512]   put this line outside the encoder for optimizing reason

            # return emb, out.transpose(0, 1).contiguous(), lengths
        elif add_noise==False:
            out = self.layer_norm(out)
            memory_bank = out.transpose(0, 1).contiguous()
            memory_bank = self.squeeze_hidden(memory_bank)  # [16,64,100]
            memory_bank = self.activation(memory_bank)

        return emb, memory_bank, lengths


    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)
