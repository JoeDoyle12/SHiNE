# pytorch_model_factory.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (batch_first=True)."""
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        if d_model % 2 == 1:
            # if odd dims, last even slot left as zero for cos
            pe[:, 1::2] = torch.cos(pos * div_term[:(d_model//2)])
        else:
            pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.dtype)
        return x

def _get_activation(act): #
    if act.lower() == 'relu':
        return nn.ReLU()
    elif act.lower() == 'gelu':
        return nn.GELU()
    elif act.lower() == 'sigmoid':
        return nn.Sigmoid()
    return nn.ReLU()

def _init_weights(module, init_linear='kaiming', init_rnn='orthogonal'):
    """Simple weight init helper applied via module.apply(fn)."""
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        if init_linear == 'kaiming':
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        elif init_linear == 'xavier':
            nn.init.xavier_uniform_(module.weight)
        elif init_linear == 'normal':
            nn.init.normal_(module.weight, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                if init_rnn == 'orthogonal':
                    nn.init.orthogonal_(param.data)
                elif init_rnn == 'xavier':
                    nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                if init_rnn == 'orthogonal':
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

class BiLSTMTransformerClassifier(nn.Module):
    """
    Bi-LSTM -> Transformer Encoder -> (pooling) -> MLP classifier / regression heads.
    """

    def __init__(
        self,
        input_size=3,                 # per-timestep raw features (e.g., TTV, TDV, sigma)
        lstm_hidden=64,               # LSTM hidden size (per direction)
        n_lstm_layers=1,              # number of stacked LSTM layers
        bidirectional=True,
        transformer_d_model=128,      # d_model for transformer (must match proj_out)
        n_transformer_layers=2,
        n_heads=8,
        transformer_ffn_dim=None,     # if None uses 4 * d_model
        transformer_dropout=0.1,
        use_positional_encoding=True,
        use_cls_token=True,           # prepend learned CLS token for pooling
        pooling='cls',                # one of 'cls', 'mean', 'max'
        second_input_size=None,       # if provided, optional per-sequence side vector
        second_fc_hidden=None,        # list or int for second-input MLP
        decoder_hidden=[128, 64],     # MLP head sizes; last layer maps to num_labels
        num_labels=1,                 # classification output size (binary -> 1 logit)
        act='ReLU',
        dropout=0.1,
        norm_type=None,               # 'layer' or 'batch' or None
        init_linear='kaiming',
        init_rnn='orthogonal',
        device=None
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.pooling = pooling
        self.use_cls_token = use_cls_token

        # ====== Bi-LSTM encoder ======
        # Use LSTM (you can switch to GRU by replacing nn.LSTM -> nn.GRU if desired)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_lstm_layers > 1 else 0.0
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1) # Bi-LSTM correction

        # project LSTM outputs to transformer d_model
        self.proj = nn.Linear(lstm_out_dim, transformer_d_model)

        # optional normalization on timestep features
        if norm_type == 'layer':
            self.input_norm = nn.LayerNorm(input_size)
        elif norm_type == 'batch':
            self.input_norm = nn.BatchNorm1d(input_size)
        else:
            self.input_norm = None

        # positional encoding
        self.use_pe = use_positional_encoding
        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(transformer_d_model)

        # CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_d_model))

        # Transformer encoder
        if transformer_ffn_dim is None:
            transformer_ffn_dim = transformer_d_model * 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=n_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=transformer_dropout,
            batch_first=True,
            activation=act.lower() if hasattr(nn, act) else 'relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        # ====== optional second-input (global metadata) ======
        self.has_second = second_input_size is not None
        if self.has_second:
            # build simple MLP for second input
            if second_fc_hidden is None:
                second_fc_hidden = [max(second_input_size, transformer_d_model // 2)]
            if isinstance(second_fc_hidden, int):
                second_fc_hidden = [second_fc_hidden]
            seq = []
            prev = second_input_size
            for h in second_fc_hidden:
                seq += [nn.Linear(prev, h), _get_activation(act), nn.Dropout(dropout)]
                prev = h
            seq += [nn.Linear(prev, transformer_d_model), _get_activation(act)]
            self.second_mlp = nn.Sequential(*seq)
        else:
            self.second_mlp = None

        # ====== classifier / decoder MLP ======
        mlp_layers = []
        prev = 3 * transformer_d_model + (transformer_d_model if self.has_second else 0) # Should change to 3 * transformer_d_model due to multi-pooling (originally transformer_d_model if cls only)
        if isinstance(decoder_hidden, int):
            decoder_hidden = [decoder_hidden]
        for h in decoder_hidden:
            mlp_layers.append(nn.Linear(prev, h))
            mlp_layers.append(_get_activation(act))
            mlp_layers.append(nn.Dropout(dropout))
            prev = h
        # final layer -> num_labels (regression or logits)
        mlp_layers.append(nn.Linear(prev, num_labels))
        mlp_layers.append(_get_activation('sigmoid')) # Last layer sigmoid, because we want something in the range 0-1
        self.classifier = nn.Sequential(*mlp_layers)

        # optional extra regression heads (dictionary of heads)
        self.regression_heads = nn.ModuleDict()  # user can add .regression_heads['incli'] = nn.Linear(prev,1)

        # ====== inits ======
        self.apply(lambda m: _init_weights(m, init_linear, init_rnn))

        # move to device
        self.to(self.device)

    def forward(self, x, x_mask=None, second_input=None):
        """
        x: (batch, seq_len, input_size)
        x_mask: Boolean mask (batch, seq_len) where True indicates padding/masked positions
        second_input: (batch, second_input_size) or None
        """
        # optional input norm
        if self.input_norm is not None:
            # if BatchNorm1d we need to transpose: (B, C, L) where C=input_size
            if isinstance(self.input_norm, nn.BatchNorm1d):
                x = x.transpose(1, 2)  # (B, C, L)
                x = self.input_norm(x)
                x = x.transpose(1, 2)
            else:
                x = self.input_norm(x)

        # Bi-LSTM
        # pack padded if mask provided for efficiency (optional)
        if x_mask is not None:
            # x_mask: True where padding. create lengths
            lengths = (~x_mask).sum(dim=1).cpu()
            print(lengths)
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)  # (B, seq_len, lstm_out_dim)

        # project to transformer d_model
        proj = self.proj(lstm_out)  # (B, seq_len, d_model)

        # optionally prepend cls token
        if self.use_cls_token:
            batch_size = proj.size(0)
            cls = self.cls_token.expand(batch_size, -1, -1)  # (B,1,d_model)
            proj = torch.cat([cls, proj], dim=1)  # seq_len+1

            # if mask exists, extend mask for cls (cls not masked)
            if x_mask is not None:
                cls_mask = torch.zeros((x_mask.size(0), 1), dtype=x_mask.dtype, device=x_mask.device)
                x_mask = torch.cat([cls_mask, x_mask], dim=1)

        # positional encoding
        if self.use_pe:
            proj = self.pos_enc(proj)

        # Transformer - note PyTorch transformer uses src_key_padding_mask with True = masked
        trans_out = self.transformer(proj, src_key_padding_mask=x_mask)

        # Multi-Pooling
        pooled_list = []

        # CLS pooling
        if self.use_cls_token:
            pooled_cls = trans_out[:, 0, :]
            pooled_list.append(pooled_cls)

        # Mean pooling
        if x_mask is not None:
            inv_mask = (~x_mask).unsqueeze(-1).float()
            summ = (trans_out * inv_mask).sum(dim=1)
            denom = inv_mask.sum(dim=1).clamp(min=1.0)
            pooled_mean = summ / denom
        else:
            pooled_mean = trans_out.mean(dim=1)
        pooled_list.append(pooled_mean)

        # Max pooling
        if x_mask is not None:
            masked = trans_out.masked_fill(x_mask.unsqueeze(-1), float('-1e9'))
            pooled_max = masked.max(dim=1)[0]
        else:
            pooled_max = trans_out.max(dim=1)[0]
        pooled_list.append(pooled_max)

        # Concatenate all pooled vectors
        pooled = torch.cat(pooled_list, dim=-1)

        # concat second input if present
        if self.has_second and second_input is not None:
            second_emb = self.second_mlp(second_input)  # (B, d_model)
            concat = torch.cat([pooled, second_emb], dim=-1)
        else:
            concat = pooled

        logits = self.classifier(concat)  # (B, num_labels)

        # regression heads dict outputs (if any) - user can add heads after instantiation
        extra = {name: head(concat) for name, head in self.regression_heads.items()}

        return logits, extra

# Factory function
def build_ttvtdv_model(
    input_size=2, # number of input features per timestep (e.g., TTV, TDV)
    lstm_hidden=64, # number of LSTM features
    n_lstm_layers=1, # number of LSTM layers
    bidirectional=True,
    transformer_d_model=128,
    n_transformer_layers=2,
    n_heads=8,
    transformer_ffn_dim=None,
    transformer_dropout=0.1,
    use_positional_encoding=True,
    use_cls_token=True,
    pooling='cls',
    second_input_size=None,
    second_fc_hidden=None,
    decoder_hidden=[128, 64],
    num_labels=1,
    act='ReLU',
    dropout=0.1,
    norm_type=None,
    init_linear='kaiming',
    init_rnn='orthogonal',
    device=None
):
    model = BiLSTMTransformerClassifier(
        input_size=input_size,
        lstm_hidden=lstm_hidden,
        n_lstm_layers=n_lstm_layers,
        bidirectional=bidirectional,
        transformer_d_model=transformer_d_model,
        n_transformer_layers=n_transformer_layers,
        n_heads=n_heads,
        transformer_ffn_dim=transformer_ffn_dim,
        transformer_dropout=transformer_dropout,
        use_positional_encoding=use_positional_encoding,
        use_cls_token=use_cls_token,
        pooling=pooling,
        second_input_size=second_input_size,
        second_fc_hidden=second_fc_hidden,
        decoder_hidden=decoder_hidden,
        num_labels=num_labels,
        act=act,
        dropout=dropout,
        norm_type=norm_type,
        init_linear=init_linear,
        init_rnn=init_rnn,
        device=device
    )
    return model

from torchviz import make_dot 
import hiddenlayer as hl
# # Example usage:
if __name__ == '__main__': # simple test of model, padding, masking, etc.
    # instantiate with defaults
    model = build_ttvtdv_model(input_size=2, lstm_hidden=64, transformer_d_model=128)
    print(model)
    # test a forward pass with fake data
    batch_size, sequence_len = 4, 50  # batch, seq_len
    x = torch.randn(batch_size, sequence_len, 2)
    print(x)
    mask = torch.zeros(batch_size, sequence_len, dtype=torch.bool)  # no padding
    mask[0][30:] = True  # example padding for first sequence
    print(mask)
    logits, extra = model(x, x_mask=mask)
    print('logits shape', logits.shape)
    print(logits)
    # make_dot(logits).render("rnn_transformer_model", format="svg")

    # torch.onnx.export(
    #     model,
    #     x,
    #     "model.onnx",
    #     export_params=True,
    #     opset_version=17,
    #     do_constant_folding=True,
    # )
