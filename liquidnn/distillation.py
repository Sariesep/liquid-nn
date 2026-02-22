"""
Knowledge Distillation — Draft Model Eğitimi

Ana modelin (teacher) soft logit'lerini kullanarak
hafif draft modeli (student) eğitir.

Loss: KL(softmax(teacher_logits/T) || softmax(student_logits/T))

Teacher frozen (eval, no grad), student backprop ile güncellenir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DistillationTrainer:
    """
    Teacher-Student knowledge distillation eğitici.

    Teacher (ana model) eval modunda çalışır, gradient almaz.
    Student (draft model) teacher'ın yumuşatılmış çıktılarını
    taklit etmeyi öğrenir.

    Args:
        teacher:        Ana model (MiniLiquidGPT)
        student:        Draft model (MiniLiquidGPT)
        lr:             Öğrenme hızı
        temperature_kd: Distillation sıcaklığı (yüksek → daha yumuşak dağılım)
        alpha:          KD loss ağırlığı (1-alpha = hard label loss ağırlığı)
        device:         Hedef cihaz
    """

    def __init__(self, teacher: nn.Module, student: nn.Module,
                 lr: float = 1e-3, temperature_kd: float = 4.0,
                 alpha: float = 0.7, device: Optional[torch.device] = None):
        self.teacher = teacher
        self.student = student
        self.temperature_kd = temperature_kd
        self.alpha = alpha
        self.device = device or next(teacher.parameters()).device

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=lr, weight_decay=1e-4
        )

    def distill_step(self, input_ids: torch.Tensor) -> dict:
        """
        Tek bir distillation adımı.

        Args:
            input_ids: [B, T] token ID'leri

        Returns:
            {'loss': float, 'kd_loss': float, 'ce_loss': float}
        """
        input_ids = input_ids.to(self.device)
        B, T = input_ids.shape

        # Teacher forward (no grad, RNN sıralı)
        self.teacher.init_hidden(B, self.device)
        self.teacher.reset_hebb()
        teacher_logits_list = []
        with torch.no_grad():
            for t in range(T):
                t_logits = self.teacher.forward_token(
                    input_ids[:, t], t, enable_plasticity=True
                )
                teacher_logits_list.append(t_logits)
        teacher_logits = torch.stack(teacher_logits_list, dim=1)  # [B, T, V]

        # Student forward
        self.student.train()
        self.student.init_hidden(B, self.device)
        self.student.reset_hebb()
        student_logits_list = []
        for t in range(T):
            s_logits = self.student.forward_token(
                input_ids[:, t], t, enable_plasticity=False
            )
            student_logits_list.append(s_logits)
        student_logits = torch.stack(student_logits_list, dim=1)  # [B, T, V]

        # KD Loss: KL divergence on softened logits
        T_kd = self.temperature_kd
        teacher_soft = F.log_softmax(teacher_logits / T_kd, dim=-1)
        student_soft = F.log_softmax(student_logits / T_kd, dim=-1)
        kd_loss = F.kl_div(
            student_soft, teacher_soft.exp(),
            reduction='batchmean'
        ) * (T_kd ** 2)

        # Hard label CE loss (next token prediction)
        # targets: input_ids shifted left by 1
        if T > 1:
            targets = input_ids[:, 1:]  # [B, T-1]
            ce_logits = student_logits[:, :-1]  # [B, T-1, V]
            ce_loss = F.cross_entropy(
                ce_logits.reshape(-1, ce_logits.size(-1)),
                targets.reshape(-1)
            )
        else:
            ce_loss = torch.tensor(0.0, device=self.device)

        # Combined loss
        loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'kd_loss': kd_loss.item(),
            'ce_loss': ce_loss.item(),
        }

    def distill_epoch(self, dataloader, verbose: bool = True) -> dict:
        """
        Bir epoch boyunca distillation.

        Args:
            dataloader: Her batch [B, T] input_ids döndüren DataLoader
            verbose: İlerleme yazdır

        Returns:
            {'avg_loss': float, 'avg_kd_loss': float, 'avg_ce_loss': float}
        """
        total_loss = 0.0
        total_kd = 0.0
        total_ce = 0.0
        n_steps = 0

        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0]
            else:
                input_ids = batch

            metrics = self.distill_step(input_ids)
            total_loss += metrics['loss']
            total_kd += metrics['kd_loss']
            total_ce += metrics['ce_loss']
            n_steps += 1

            if verbose and (batch_idx + 1) % 10 == 0:
                avg = total_loss / n_steps
                print(f"  Step {batch_idx+1}: loss={avg:.4f}")

        n_steps = max(n_steps, 1)
        return {
            'avg_loss': total_loss / n_steps,
            'avg_kd_loss': total_kd / n_steps,
            'avg_ce_loss': total_ce / n_steps,
        }
