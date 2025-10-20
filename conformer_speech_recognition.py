import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, Audio, concatenate_datasets
import sentencepiece as spm
from jiwer import wer
from tqdm import tqdm
import json
import math
import random
import numpy as np
import heapq

SAMPLE_RATE = 16000
N_MELS = 80
HOP_LENGTH = 160
N_FFT = 400
D_MODEL = 144
NUM_HEADS = 4
FFN_DIM = 576
NUM_CONFORMER_BLOCKS = 12
CONV_KERNEL_SIZE = 31
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
WARMUP_STEPS = 10000
SP_VOCAB_SIZE = 5000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DROPOUT = 0.1
GRAD_CLIP = 5.0

FREQ_MASK_PARAM = 27
NUM_FREQ_MASKS = 2
TIME_MASK_PARAM = 100
NUM_TIME_MASKS = 10

SPEED_PERTURB = [0.9, 1.0, 1.1]

print(f"Device: {DEVICE}")
print(f"Model: D={D_MODEL}, H={NUM_HEADS}, Blocks={NUM_CONFORMER_BLOCKS}, FFN={FFN_DIM}")

def load_librispeech_half():
    train_ds = load_dataset("librispeech_asr", "clean", split="train.100+train.360")
    val_ds = load_dataset("librispeech_asr", "clean", split="validation")
    test_ds = load_dataset("librispeech_asr", "clean", split="test")

    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    val_ds = val_ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    test_ds = test_ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds

def train_sentencepiece(train_dataset, prefix="spm"):
    if os.path.exists(f'{prefix}.model'):
        sp = spm.SentencePieceProcessor()
        sp.load(f'{prefix}.model')
        return sp
    
    texts = []
    for item in tqdm(train_dataset, desc="Extracting texts"):
        text = item['text'].lower().strip()
        if text:
            texts.append(text)
    
    with open("train_texts.txt", "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")
    
    spm.SentencePieceTrainer.train(
        input="train_texts.txt",
        model_prefix=prefix,
        vocab_size=SP_VOCAB_SIZE,
        character_coverage=1.0,
        model_type="unigram",
        pad_id=0,
        unk_id=1,
        bos_id=-1,
        eos_id=-1
    )
    
    sp = spm.SentencePieceProcessor()
    sp.load(f'{prefix}.model')
    print(f"Vocab size: {sp.get_piece_size()}")
    return sp

mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    power=2.0
)

def compute_log_mel(waveform):
    mel = mel_transform(waveform)
    log_mel = torch.log(mel + 1e-9)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    return log_mel.transpose(-2, -1)

def apply_specaugment(spec):
    spec = spec.transpose(0, 1)
    
    for _ in range(NUM_FREQ_MASKS):
        spec = T.FrequencyMasking(freq_mask_param=FREQ_MASK_PARAM)(spec)
    
    for _ in range(NUM_TIME_MASKS):
        spec = T.TimeMasking(time_mask_param=TIME_MASK_PARAM)(spec)
    
    return spec.transpose(0, 1)

def speed_perturbation(waveform, speed):
    if speed == 1.0:
        return waveform
    
    new_sr = int(SAMPLE_RATE * speed)
    resampler = T.Resample(SAMPLE_RATE, new_sr)
    perturbed = resampler(waveform)
    
    resampler_back = T.Resample(new_sr, SAMPLE_RATE)
    return resampler_back(perturbed)

class LibriSpeechDataset(Dataset):
    def __init__(self, hf_dataset, sp, augment=False):
        self.dataset = hf_dataset
        self.sp = sp
        self.augment = augment
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        waveform = torch.tensor(item['audio']['array'], dtype=torch.float32)
        
        if self.augment:
            speed = random.choice(SPEED_PERTURB)
            waveform = speed_perturbation(waveform, speed)
        
        if len(waveform) < 1600:
            waveform = F.pad(waveform, (0, 1600 - len(waveform)))
        
        log_mel = compute_log_mel(waveform)
        
        if self.augment:
            log_mel = apply_specaugment(log_mel)
        
        text = item['text'].lower().strip()
        tokens = self.sp.encode(text, out_type=int)
        if not tokens:
            tokens = [1]
        
        labels = torch.tensor(tokens, dtype=torch.long)
        return log_mel, labels, text

def collate_fn(batch):
    specs, labels, texts = zip(*batch)
    
    spec_lengths = torch.tensor([len(spec) for spec in specs])
    padded_specs = nn.utils.rnn.pad_sequence(specs, batch_first=True)
    
    label_lengths = torch.tensor([len(label) for label in labels])
    padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    
    return padded_specs, spec_lengths, padded_labels, label_lengths, texts

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class FeedForwardModule(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = self.w1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.w2(x)
        x = self.dropout2(x)
        return residual + 0.5 * x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.ln(x)
        x, _ = self.attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        return residual + x

class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pw_conv_1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = Swish()
        self.pw_conv_2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = x.transpose(1, 2)
        x = self.pw_conv_1(x)
        x = self.glu(x)
        x = self.dw_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pw_conv_2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return residual + x

class ConformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, kernel_size, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, d_ff, dropout)
        self.mhsa = MultiHeadAttention(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, d_ff, dropout)
        self.ln = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = self.ffn1(x)
        x = self.mhsa(x)
        x = self.conv(x)
        x = self.ffn2(x)
        return self.ln(x)

class ConformerCTC(nn.Module):
    def __init__(self, n_mels, vocab_size):
        super().__init__()
        
        self.conv_subsampling = nn.Sequential(
            nn.Conv2d(1, D_MODEL, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(D_MODEL, D_MODEL, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.linear_proj = nn.Linear(D_MODEL * (n_mels // 4), D_MODEL)
        self.pos_encoding = PositionalEncoding(D_MODEL)
        
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(D_MODEL, NUM_HEADS, FFN_DIM, CONV_KERNEL_SIZE, DROPOUT)
            for _ in range(NUM_CONFORMER_BLOCKS)
        ])
        
        self.final_ln = nn.LayerNorm(D_MODEL)
        self.ctc_head = nn.Linear(D_MODEL, vocab_size)
        
        nn.init.xavier_uniform_(self.ctc_head.weight)
        nn.init.constant_(self.ctc_head.bias, 0.0)
        self.ctc_head.bias.data[0] = -3.0
        
    def forward(self, x):
        B, T, _ = x.size()
        x = x.unsqueeze(1)
        x = self.conv_subsampling(x)
        B, C, T_sub, F_sub = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, T_sub, C * F_sub)
        x = self.linear_proj(x)
        x = self.pos_encoding(x)
        
        for block in self.conformer_blocks:
            x = block(x)
        
        x = self.final_ln(x)
        x = self.ctc_head(x)
        
        return F.log_softmax(x, dim=-1)

def beam_search_decode(log_probs, sp, beam_width=10, blank_id=0):
    T, V = log_probs.shape
    log_probs = log_probs.cpu().numpy()
    beams = {(): (0.0, -np.inf)}
    
    for t in range(T):
        new_beams = {}
        
        for prefix, (pb, pnb) in beams.items():
            pnb_combined = np.logaddexp(pb, pnb)
            
            for c in range(V):
                prob = log_probs[t, c]
                
                if c == blank_id:
                    new_pb = np.logaddexp(pb + prob, pnb_combined + prob)
                    key = prefix
                    
                    if key not in new_beams:
                        new_beams[key] = (new_pb, -np.inf)
                    else:
                        new_beams[key] = (np.logaddexp(new_beams[key][0], new_pb), new_beams[key][1])
                else:
                    if prefix and prefix[-1] == c:
                        new_pnb = pb + prob
                    else:
                        new_pnb = pnb_combined + prob
                    
                    key = prefix + (c,)
                    if key not in new_beams:
                        new_beams[key] = (-np.inf, new_pnb)
                    else:
                        new_beams[key] = (new_beams[key][0], np.logaddexp(new_beams[key][1], new_pnb))
        
        beams = dict(sorted(
            new_beams.items(),
            key=lambda x: np.logaddexp(x[1][0], x[1][1]),
            reverse=True
        )[:beam_width])
    
    if beams:
        best_prefix = max(beams.items(), key=lambda x: np.logaddexp(x[1][0], x[1][1]))[0]
        if best_prefix:
            try:
                return sp.decode(list(best_prefix))
            except:
                return ""
    
    return ""

def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(loader, desc="Training")
    for specs, spec_lens, labels, label_lens, _ in pbar:
        specs = specs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        log_probs = model(specs)
        log_probs = log_probs.transpose(0, 1)
        input_lengths = (spec_lens // 4).clamp(min=1).to(device)
        target_lengths = label_lens.to(device)
        
        if not (input_lengths >= target_lengths).all():
            continue
        
        try:
            loss = criterion(log_probs, labels, input_lengths, target_lengths)
            if torch.isfinite(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        except:
            continue
    
    return total_loss / max(num_batches, 1)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for specs, spec_lens, labels, label_lens, _ in tqdm(loader, desc="Validation"):
            specs = specs.to(device)
            labels = labels.to(device)
            log_probs = model(specs)
            log_probs = log_probs.transpose(0, 1)
            input_lengths = (spec_lens // 4).clamp(min=1).to(device)
            target_lengths = label_lens.to(device)
            
            if not (input_lengths >= target_lengths).all():
                continue
            
            try:
                loss = criterion(log_probs, labels, input_lengths, target_lengths)
                if torch.isfinite(loss):
                    total_loss += loss.item()
                    num_batches += 1
            except:
                continue
    
    return total_loss / max(num_batches, 1)

def evaluate(model, loader, sp, device, use_beam_search=True):
    model.eval()
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for specs, _, _, _, texts in tqdm(loader, desc="Evaluating"):
            specs = specs.to(device)
            log_probs = model(specs)
            
            for i in range(len(specs)):
                if use_beam_search:
                    pred_text = beam_search_decode(log_probs[i], sp, beam_width=10)
                
                all_predictions.append(pred_text)
                all_references.append(texts[i].lower())
    
    word_error_rate = wer(all_references, all_predictions)
    
    print("Sample Predictions:")
    for i in range(min(10, len(all_predictions))):
        print(f"\nRef: {all_references[i][:60]}")
        print(f"Hyp: {all_predictions[i][:60]}")
    
    return word_error_rate

def main():
    train_dataset, val_dataset, test_dataset = load_librispeech_half()
    
    sp = train_sentencepiece(train_dataset)
    
    train_data = LibriSpeechDataset(train_dataset, sp, augment=True)
    val_data = LibriSpeechDataset(val_dataset, sp, augment=False)
    test_data = LibriSpeechDataset(test_dataset, sp, augment=False)
    
    train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_data, BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    model = ConformerCTC(n_mels=N_MELS, vocab_size=sp.get_piece_size()).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {total_params/1e6:.2f}M\n")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=1e-6
    )
    
    total_steps = len(train_loader) * EPOCHS
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, 'best_model.pt')
        
        torch.save(model.state_dict(), f'checkpoint_ep_{epoch}.pt')
    
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_wer = evaluate(model, test_loader, sp, DEVICE, use_beam_search=True)
    print(f"\nTest WER: {test_wer*100:.2f}%")
    
    results = {
        'test_wer': float(test_wer),
        'best_epoch': int(checkpoint['epoch']),
        'best_val_loss': float(checkpoint['val_loss']),
        'total_params': int(total_params),
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
if __name__ == "__main__":
    main()