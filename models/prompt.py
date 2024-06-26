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
        self.share_prompt = config['prompt']['share_prompt']
        self.num_prompts = config['clip']['num_prompts']
        if self.num_prompts!=0:
            if self.share_prompt:
                self.prompt = nn.Parameter(
                    torch.empty(1, self.num_prompts, 640, dtype=torch.float32),
                    requires_grad=True)
                nn.init.normal_(self.prompt, std=0.02)
            else:
                self.norm_prompt = nn.Parameter(
                    torch.empty(1, self.num_prompts, 640, dtype=torch.float32),
                    requires_grad=True)
                nn.init.normal_(self.norm_prompt, std=0.02)
                self.anorm_prompt = nn.Parameter(
                    torch.empty(1, self.num_prompts, 640, dtype=torch.float32),
                    requires_grad=True)
                nn.init.normal_(self.anorm_prompt, std=0.02)

    def forward(self, x=None, normalize=False):
        norm_text = f'normal {self.class_name}'
        anorm_text = f'damaged {self.class_name}'
        norm_embs = self.encode_text(norm_text, normalize)
        anorm_embs = self.encode_text(anorm_text, normalize)
        return norm_embs, anorm_embs

    def encode_text(self, text, normalize=False):
        cast_dtype = self.clip.transformer.get_cast_dtype()
        text_length = len(text.split(' '))
        text_token = tokenize(text).to('cuda')
        x = self.clip.token_embedding(text_token).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        if self.num_prompts!=0:
            # adding context
            sos = x[:, 0][:, None]
            class_embs = x[:, 1:text_length + 1]
            eos = x[:, text_length + 1][:, None]
            pad = x[:, text_length + 1:]
            if self.share_prompt:
                prompt = self.prompt.to(x.device)
                x = torch.cat([sos, prompt, class_embs, eos, pad], dim=1)
            else:
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


class Learned_Object(nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip

        self.prompt_linear = nn.Sequential(
            nn.Linear(640, 64),
            nn.ReLU(),
            nn.Linear(64, 640))


        self.norm_prompt = 'a photo of flawless objects'
        self.anorm_prompt = 'a photo of damaged objects'

    def forward(self, x):
        object_embs = self.prompt_linear(x) #(N, d)
        #object_embs = torch.zeros_like(x).cuda()
        object_embs = object_embs[:, None]  #(N, 1, d)
        norm_embs = self.encode_text(self.norm_prompt, object_embs)
        anorm_embs = self.encode_text(self.anorm_prompt, object_embs)
        return torch.stack([norm_embs, anorm_embs], dim=1) #(N, 2, d)

    def encode_text(self, prompt, object_embs): #(N, 1, d)
        N = object_embs.shape[0]
        cast_dtype = self.clip.transformer.get_cast_dtype()

        text_length = len(prompt.split(' '))
        text_token = tokenize(prompt).to('cuda') #[<sos> prompt text <eos>, 0, 0, 0,....]
        x = self.clip.token_embedding(text_token).to(cast_dtype)  # [1, n_ctx, d_model] (1, 1+2+74, 640)
        # adding object
        sos = x[:, 0][:, None].repeat(N, 1, 1) #<sos>
        prompt = x[:, 1:text_length].repeat(N, 1, 1) #a photo of flawless
        object_prompt = x[:, text_length:text_length+1].repeat(N, 1, 1) + object_embs #objects
        eos = x[:, text_length+1][:, None].repeat(N, 1, 1) #<eos>
        pad = x[:, text_length+1:].repeat(N, 1, 1) #<padding>
        x = torch.cat([sos, prompt, object_prompt, eos, pad], dim=1)  #(1, 5+1+2+70, 640)
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
                x = x @ self.clip.text_projection #(1, 640)
        return x




