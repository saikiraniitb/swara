"""N2: tiny causal NeuCodec LM with Swara text memory."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class N2Config:
    linguistic_vocab_size: int
    d_model: int = 128
    heads: int = 4
    ffn_dim: int = 512
    text_layers: int = 1
    decoder_layers: int = 1
    max_text_tokens: int = 512
    max_frames: int = 4096
    codec_cardinality: int = 65536
    bos_id: int = 65536
    eos_id: int = 65537

class N2Model(nn.Module):
    def __init__(self, config: N2Config):
        super().__init__(); self.config=config
        if config.bos_id <= config.codec_cardinality-1 or config.eos_id <= config.bos_id:
            raise ValueError('special acoustic IDs must be outside codec IDs')
        self.codec_embedding=nn.Embedding(config.eos_id+1, config.d_model)
        self.linguistic_embedding=nn.Embedding(config.linguistic_vocab_size, config.d_model)
        self.text_position=nn.Embedding(config.max_text_tokens, config.d_model)
        enc_layer=nn.TransformerEncoderLayer(config.d_model,config.heads,config.ffn_dim,batch_first=True,norm_first=True)
        self.text_encoder=nn.TransformerEncoder(enc_layer,config.text_layers)
        dec_layer=nn.TransformerDecoderLayer(config.d_model,config.heads,config.ffn_dim,batch_first=True,norm_first=True)
        self.decoder=nn.TransformerDecoder(dec_layer,config.decoder_layers)
        self.text_norm=nn.LayerNorm(config.d_model); self.decoder_norm=nn.LayerNorm(config.d_model)
        self.audio_position=nn.Embedding(config.max_frames,config.d_model)

    def encode_text(self, linguistic_ids):
        b,n=linguistic_ids.shape
        if n>self.config.max_text_tokens: raise ValueError('text length exceeds max_text_tokens')
        p=torch.arange(n,device=linguistic_ids.device)
        x=self.linguistic_embedding(linguistic_ids)+self.text_position(p)[None,:,:]
        return self.text_norm(self.text_encoder(x))

    def forward(self, linguistic_ids, input_tokens, targets=None):
        b,t=input_tokens.shape
        if t>self.config.max_frames: raise ValueError('frame length exceeds max_frames')
        memory=self.encode_text(linguistic_ids)
        p=torch.arange(t,device=input_tokens.device)
        x=self.codec_embedding(input_tokens)+self.audio_position(p)[None,:,:]
        mask=torch.triu(torch.ones(t,t,device=x.device,dtype=torch.bool),diagonal=1)
        h=self.decoder(x,memory,tgt_mask=mask)
        h=self.decoder_norm(h)
        logits=F.linear(h,self.codec_embedding.weight[:self.config.codec_cardinality])
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1))
        return logits,loss,h

    @torch.no_grad()
    def generate(self, linguistic_ids, frames, temperature=0.0):
        self.eval(); out=torch.full((linguistic_ids.shape[0],1),self.config.bos_id,dtype=torch.long,device=linguistic_ids.device)
        for _ in range(frames):
            logits,_,_=self.forward(linguistic_ids,out)
            nxt=logits[:,-1,:]
            if temperature and temperature>0: nxt=torch.multinomial(torch.softmax(nxt/temperature,-1),1).squeeze(-1)
            else: nxt=nxt.argmax(-1)
            out=torch.cat([out,nxt[:,None]],1)
        return out[:,1:]

def parameter_count(model): return sum(p.numel() for p in model.parameters())
