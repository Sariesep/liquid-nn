"""Quick training test — verifies backward pass works."""
import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from liquidnn import MiniLiquidGPT
import torch.nn.functional as F

m = MiniLiquidGPT(
    vocab_size=100, embed_dim=64, num_fast=1, num_deep=1,
    use_attention=False,
    use_neuromod=True, use_homeostasis=True,
    use_dual_hebb=True, use_consolidation=True, use_rmsnorm=True,
)
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
x = torch.randint(0, 100, (2, 32))
y = torch.randint(0, 100, (2, 32))

for epoch in range(5):
    logits = m(x, chunk_size=8)
    loss = F.cross_entropy(logits.view(-1, 100), y.view(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"  Epoch {epoch+1}: loss={loss.item():.4f}")
    m.init_hidden(2, x.device)
    m.reset_hebb()

print("\n🏆 Training loop OK!")
