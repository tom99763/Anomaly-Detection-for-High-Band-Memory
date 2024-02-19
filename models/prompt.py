import torch
import torch.nn as nn
from open_clip.tokenizer import tokenize
import torch.nn.functional as F

def text_global_pool(x, text= None, pool_type='argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x
    return pooled, tokens


class Learned_Prompt(nn.Module):
    def __init__(self, config, clip):
        super().__init__()
        self.class_name = config['clip']['class_name']
        self.clip = clip
        self.norm_prompt = nn.Parameter(
            torch.empty(1, config['clip']['num_prompts'], 640, dtype=torch.float32),
            requires_grad=True)
        nn.init.normal_(self.norm_prompt, std=0.02)
        self.anorm_prompt = nn.Parameter(
            torch.empty(1, config['clip']['num_prompts'], 640, dtype=torch.float32),
            requires_grad=True)
        nn.init.normal_(self.anorm_prompt, std=0.02)

    def forward(self, x=None, normalize=False):
        norm_text = f'green {self.class_name}'
        anorm_text = f'damaged green {self.class_name}'
        norm_embs = self.encode_text(norm_text, normalize)
        anorm_embs = self.encode_text(anorm_text, normalize)
        return norm_embs, anorm_embs

    def encode_text(self, text, normalize=False):
        cast_dtype = self.clip.transformer.get_cast_dtype()
        text_length = len(text.split(' '))
        text_token = tokenize(text)
        x = self.clip.token_embedding(text_token).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        # adding context
        sos = x[:, 0][:, None]
        class_embs = x[:, 1:text_length + 1]
        eos = x[:, text_length + 1][:, None]
        pad = x[:, text_length + 1:]
        if 'damaged' in text:
            anorm_prompt = self.anorm_prompt.to(x.device)
            x = torch.cat([sos, anorm_prompt, class_embs, eos, pad], dim=1)
        else:
            norm_prompt = self.norm_prompt.to(x.device)
            x = torch.cat([sos, norm_prompt, class_embs, eos, pad], dim=1)
        x = x[:, :77]

        # forward
        x = x + self.clip.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x, attn_mask=self.clip.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, _ = text_global_pool(x, text_token, self.clip.text_pool_type)
        if self.clip.text_projection is not None:
            if isinstance(self.clip.text_projection, nn.Linear):
                x = self.clip.text_projection(x)
            else:
                x = x @ self.clip.text_projection
        return F.normalize(x, dim=-1) if normalize else x