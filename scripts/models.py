import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2, use_batch_norm=True):
        super(ResidualBlock, self).__init__()
        self.use_batch_norm = use_batch_norm
        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        
        if use_batch_norm:
            self.norm1 = nn.BatchNorm1d(hidden_size)
            self.norm2 = nn.BatchNorm1d(input_size)
    
    def forward(self, x):
        residual = x
        
        out = self.linear1(x)
        if self.use_batch_norm:
            out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        if self.use_batch_norm:
            out = self.norm2(out)
        
        out += residual
        out = self.activation(out)
        
        return out

class EmbeddingEncoder(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.2, use_batch_norm=True, num_residual_blocks=3):
        super(EmbeddingEncoder, self).__init__()
        
        hidden_size = max(input_size, output_size * 2)
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, hidden_size * 2, dropout, use_batch_norm)
            for _ in range(num_residual_blocks)
        ])
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.input_norm = nn.BatchNorm1d(input_size)
            self.output_norm = nn.BatchNorm1d(output_size)
    
    def forward(self, x):
        if self.use_batch_norm:
            x = self.input_norm(x)
        
        x = self.input_projection(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.output_projection(x)
        
        if self.use_batch_norm:
            x = self.output_norm(x)
        
        return x

class AttentionFusion(nn.Module):
    """N-way attention fusion. Supports 2+ modalities."""
    def __init__(self, feature_size, num_modalities=2, dropout=0.2):
        super(AttentionFusion, self).__init__()
        self.num_modalities = num_modalities
        self.attention = nn.Sequential(
            nn.Linear(feature_size * num_modalities, feature_size),
            nn.Tanh(),
            nn.Linear(feature_size, num_modalities),
            nn.Softmax(dim=-1)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, *features):
        combined = torch.cat(features, dim=-1)
        attention_weights = self.attention(combined)  # [B, N]
        
        fused = sum(f * attention_weights[:, i:i+1] for i, f in enumerate(features))
        return self.dropout(fused), attention_weights
    
class BaseClassifier(nn.Module):
    def __init__(self, hidden_size=256, num_classes=10, emb_size_audio=0, emb_size_text=0, 
                 dropout=0.2, use_batch_norm=True, mode="both", num_residual_blocks=3, 
                 use_attention_fusion=True, num_top_classes=0, emb_size_aux=0):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_top_classes = num_top_classes
        self.emb_size_audio = emb_size_audio
        self.emb_size_text = emb_size_text
        self.emb_size_aux = emb_size_aux
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.mode = mode
        self.num_residual_blocks = num_residual_blocks
        self.use_attention_fusion = use_attention_fusion and mode == "both"
        
        if self.mode in ["audio", "both"]:
            self.audio_emb_extractor = EmbeddingEncoder(
                input_size=emb_size_audio,
                output_size=hidden_size,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                num_residual_blocks=num_residual_blocks
            )
        else:
            self.audio_emb_extractor = None
        
        if self.mode in ["text", "both"]:
            self.text_emb_extractor = EmbeddingEncoder(
                input_size=emb_size_text,
                output_size=hidden_size,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                num_residual_blocks=num_residual_blocks
            )
        else:
            self.text_emb_extractor = None
        
        # Auxiliary encoder (e.g. Whisper, ConvNeXt)
        if emb_size_aux > 0 and self.mode == "both":
            self.aux_emb_extractor = EmbeddingEncoder(
                input_size=emb_size_aux,
                output_size=hidden_size,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                num_residual_blocks=num_residual_blocks
            )
        else:
            self.aux_emb_extractor = None
        
        # Count modalities for fusion
        num_modalities = sum([
            self.audio_emb_extractor is not None,
            self.text_emb_extractor is not None,
            self.aux_emb_extractor is not None,
        ])
        
        if self.mode == "both" and num_modalities >= 2:
            if self.use_attention_fusion:
                combined_size = hidden_size
                self.fusion = AttentionFusion(hidden_size, num_modalities, dropout)
            else:
                combined_size = hidden_size * num_modalities
        else:
            combined_size = hidden_size
        
        self.latent_projector = nn.Sequential(
            nn.Linear(combined_size, hidden_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout / 2)
        )
        
        self.residual_classifier = nn.ModuleList([
            ResidualBlock(hidden_size // 2, hidden_size, dropout / 2, use_batch_norm)
            for _ in range(2)
        ])
        
        self.class_predictor = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LeakyReLU(),
            nn.Dropout(dropout / 4),
            nn.Linear(hidden_size // 4, num_classes)
        )

        # Hierarchical top-class prediction head (optional)
        if num_top_classes > 0:
            self.top_class_predictor = nn.Sequential(
                nn.Linear(hidden_size // 2, num_top_classes)
            )
        else:
            self.top_class_predictor = None
    
    def forward(self, audio_emb=None, text_emb=None, aux_emb=None):
        features = []
        
        if self.audio_emb_extractor is not None and audio_emb is not None:
            audio_features = self.audio_emb_extractor(audio_emb)
            features.append(audio_features)
        
        if self.text_emb_extractor is not None and text_emb is not None:
            text_features = self.text_emb_extractor(text_emb)
            features.append(text_features)
        
        if self.aux_emb_extractor is not None and aux_emb is not None:
            aux_features = self.aux_emb_extractor(aux_emb)
            features.append(aux_features)
        
        if len(features) > 1:
            if self.use_attention_fusion:
                combined_features, attn_scores = self.fusion(*features)
            else:
                combined_features = torch.cat(features, dim=-1)
                attn_scores = None
        else:
            combined_features = features[0]
            attn_scores = None
        
        z = self.latent_projector(combined_features)
        
        for block in self.residual_classifier:
            z = block(z)
        
        class_logit = self.class_predictor(z)

        top_class_logit = self.top_class_predictor(z) if self.top_class_predictor is not None else None
        
        return z, class_logit, top_class_logit, attn_scores


if __name__ == "__main__":
    # Test 2-modality (backward compatible)
    model = BaseClassifier(
        hidden_size=128, 
        num_classes=23, emb_size_audio=512, emb_size_text=512, 
        dropout=0.2, use_batch_norm=False, mode="both", num_top_classes=5)
    audio = torch.randn(1, 512)  
    text = torch.randn(1, 512)  
    z, class_logit, top_class_logit, attn_scores = model(audio_emb=audio, text_emb=text)
    print("=== 2-Modality ===")
    print("Latent representation shape:", z.shape)
    print("Class logit shape:", class_logit.shape)
    print("Attention scores shape:", attn_scores.shape if attn_scores is not None else None)
    print("Model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Test 3-modality
    model3 = BaseClassifier(
        hidden_size=128,
        num_classes=23, emb_size_audio=512, emb_size_text=512, emb_size_aux=512,
        dropout=0.2, use_batch_norm=False, mode="both", num_top_classes=5)
    aux = torch.randn(1, 512)
    z3, cl3, tcl3, attn3 = model3(audio_emb=audio, text_emb=text, aux_emb=aux)
    print("\n=== 3-Modality ===")
    print("Latent representation shape:", z3.shape)
    print("Class logit shape:", cl3.shape)
    print("Attention scores shape:", attn3.shape if attn3 is not None else None)
    print("Model parameters:", sum(p.numel() for p in model3.parameters() if p.requires_grad))
