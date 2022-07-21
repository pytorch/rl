import torch.nn as nn


class TokensToQKV(nn.Module):
    def __init__(self, to_dim, from_dim, latent_dim):
        super().__init__()
        self.q = nn.Linear(to_dim, latent_dim)
        self.k = nn.Linear(from_dim, latent_dim)
        self.v = nn.Linear(from_dim, latent_dim)

    def forward(self, X_to, X_from):
        Q = self.q(X_to)
        K = self.k(X_from)
        V = self.v(X_from)
        return Q, K, V


class SplitHeads(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, Q, K, V):
        batch_size, to_num, latent_dim = Q.shape
        _, from_num, _ = K.shape
        d_tensor = latent_dim // self.num_heads
        Q = Q.reshape(batch_size, to_num, self.num_heads, d_tensor).transpose(1, 2)
        K = K.reshape(batch_size, from_num, self.num_heads, d_tensor).transpose(1, 2)
        V = V.reshape(batch_size, from_num, self.num_heads, d_tensor).transpose(1, 2)
        return Q, K, V


class Attention(nn.Module):
    def __init__(self, latent_dim, to_dim):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Linear(latent_dim, to_dim)

    def forward(self, Q, K, V):
        batch_size, n_heads, to_num, d_in = Q.shape
        attn = self.softmax(Q @ K.transpose(2, 3) / d_in)
        out = attn @ V
        out = self.out(out.transpose(1, 2).reshape(batch_size, to_num, n_heads * d_in))
        return out, attn


class SkipLayerNorm(nn.Module):
    def __init__(self, to_len, to_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm((to_len, to_dim))

    def forward(self, x_0, x_1):
        return self.layer_norm(x_0 + x_1)


class FFN(nn.Module):
    def __init__(self, to_dim, hidden_dim, dropout_rate=0.2):
        super().__init__()
        self.FFN = nn.Sequential(
            nn.Linear(to_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, to_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, X):
        return self.FFN(X)


class AttentionBlock(nn.Module):
    def __init__(self, to_dim, to_len, from_dim, latent_dim, num_heads):
        super().__init__()
        self.tokens_to_qkv = TokensToQKV(to_dim, from_dim, latent_dim)
        self.split_heads = SplitHeads(num_heads)
        self.attention = Attention(latent_dim, to_dim)
        self.skip = SkipLayerNorm(to_len, to_dim)

    def forward(self, X_to, X_from):
        Q, K, V = self.tokens_to_qkv(X_to, X_from)
        Q, K, V = self.split_heads(Q, K, V)
        out, attention = self.attention(Q, K, V)
        out = self.skip(X_to, out)
        return out


class EncoderTransformerBlock(nn.Module):
    def __init__(self, to_dim, to_len, latent_dim, num_heads):
        super().__init__()
        self.attention_block = AttentionBlock(
            to_dim, to_len, to_dim, latent_dim, num_heads
        )
        self.FFN = FFN(to_dim, 4 * to_dim)
        self.skip = SkipLayerNorm(to_len, to_dim)

    def forward(self, X_to):
        X_to = self.attention_block(X_to, X_to)
        X_out = self.FFN(X_to)
        return self.skip(X_out, X_to)


class DecoderTransformerBlock(nn.Module):
    def __init__(self, to_dim, to_len, from_dim, latent_dim, num_heads):
        super().__init__()
        self.attention_block = AttentionBlock(
            to_dim, to_len, from_dim, latent_dim, num_heads
        )
        self.encoder_block = EncoderTransformerBlock(
            to_dim, to_len, latent_dim, num_heads
        )

    def forward(self, X_to, X_from):
        X_to = self.attention_block(X_to, X_from)
        X_to = self.encoder_block(X_to)
        return X_to


class TransformerEncoder(nn.Module):
    def __init__(self, num_blocks, to_dim, to_len, latent_dim, num_heads):
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                EncoderTransformerBlock(to_dim, to_len, latent_dim, num_heads)
                for i in range(num_blocks)
            ]
        )

    def forward(self, X_to):
        for i in range(len(self.encoder)):
            X_to = self.encoder[i](X_to)
        return X_to


class TransformerDecoder(nn.Module):
    def __init__(self, num_blocks, to_dim, to_len, from_dim, latent_dim, num_heads):
        super().__init__()
        self.decoder = nn.ModuleList(
            [
                DecoderTransformerBlock(to_dim, to_len, from_dim, latent_dim, num_heads)
                for i in range(num_blocks)
            ]
        )

    def forward(self, X_to, X_from):
        for i in range(len(self.decoder)):
            X_to = self.decoder[i](X_to, X_from)
        return X_to


class Transformer(nn.Module):
    def __init__(
        self, num_blocks, to_dim, to_len, from_dim, from_len, latent_dim, num_heads
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_blocks, to_dim, to_len, latent_dim, num_heads
        )
        self.decoder = TransformerDecoder(
            num_blocks, from_dim, from_len, to_dim, latent_dim, num_heads
        )

    def forward(self, X_to, X_from):
        X_to = self.encoder(X_to)
        X_out = self.decoder(X_from, X_to)
        return X_out
