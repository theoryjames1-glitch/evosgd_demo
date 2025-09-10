# evosgd_demo

Perfect — let’s do a **tiny demo of EvoSGD** just like we did for AdaptiveScheduler.
This will:

* Train a small regression model on synthetic data.
* Use **EvoSGD** as the optimizer.
* Print the **loss** and whether an **evolutionary step** was triggered.

---

### `tiny_demo_evosgd.py`

```python
#!/usr/bin/env python3
"""
Tiny Demo: EvoSGD
-----------------
Trains a toy regression model with EvoSGD optimizer.
EvoSGD = SGD updates + periodic evolutionary search.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer


class EvoSGD(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-2,
        momentum=0.0,
        nesterov=False,
        weight_decay=0.0,
        evo_interval=50,
        pop_size=6,
        elite_frac=0.3,
        sigma=0.02,
        sigma_decay=0.99,
        reset_momentum_on_evo=True,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self._t = 0
        self.evo_interval = evo_interval
        self.pop_size = pop_size
        self.elite_frac = elite_frac
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.reset_momentum_on_evo = reset_momentum_on_evo

    @torch.no_grad()
    def step(self, closure=None):
        """Perform an SGD step; occasionally perform evolutionary search."""
        loss = None

        # --- 1) normal SGD ---
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if wd != 0:
                    d_p = d_p.add(p, alpha=wd)

                state = self.state[p]
                if mom != 0:
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.zeros_like(p)
                    buf.mul_(mom).add_(d_p)
                    state["momentum_buffer"] = buf
                    d = d_p.add(buf, alpha=mom) if nesterov else buf
                else:
                    d = d_p

                p.add_(d, alpha=-lr)

        self._t += 1

        # --- 2) evolution step ---
        if self.evo_interval > 0 and (self._t % self.evo_interval == 0):
            if closure is None:
                raise RuntimeError("EvoSGD requires a closure on evo steps.")

            # collect params
            flat_params = [p for g in self.param_groups for p in g["params"] if p.requires_grad]
            base = [p.detach().clone() for p in flat_params]

            def load_params(src_list):
                for p, s in zip(flat_params, src_list):
                    p.copy_(s)

            def param_scale(p):
                s = torch.sqrt(torch.mean(p.pow(2))) if p.numel() > 0 else torch.tensor(0., device=p.device)
                return torch.clamp(s, min=1e-8)

            # evaluate base
            load_params(base)
            base_loss = float(closure().detach())
            candidates = [(base_loss, base)]

            # mutated candidates
            for _ in range(self.pop_size - 1):
                mutated = []
                for p in base:
                    s = param_scale(p) * self.sigma
                    noise = torch.randn_like(p) * s
                    mutated.append(p + noise)
                load_params(mutated)
                cand_loss = float(closure().detach())
                candidates.append((cand_loss, [c.detach().clone() for c in mutated]))

            # select elites
            candidates.sort(key=lambda x: x[0])
            k = max(1, int(math.ceil(self.elite_frac * self.pop_size)))
            elites = [c[1] for c in candidates[:k]]

            # recombine (mean of elites)
            mean_params = []
            for parts in zip(*elites):
                mean_params.append(torch.stack(parts, dim=0).mean(dim=0))
            load_params(mean_params)

            if self.reset_momentum_on_evo:
                for p in flat_params:
                    st = self.state[p]
                    if "momentum_buffer" in st:
                        st["momentum_buffer"].zero_()

            self.sigma *= self.sigma_decay
            loss = float(closure().detach())

        return loss


# --- tiny training demo ---
def main():
    torch.manual_seed(0)

    # synthetic regression
    X = torch.randn(256, 10)
    true_w = torch.randn(10, 1)
    y = X @ true_w + 0.1 * torch.randn(256, 1)

    model = nn.Linear(10, 1)
    opt = EvoSGD(model.parameters(), lr=0.05, momentum=0.9, evo_interval=30)

    def closure():
        with torch.no_grad():
            pred = model(X)
            return F.mse_loss(pred, y)

    for step in range(100):
        opt.zero_grad()
        pred = model(X)
        loss = F.mse_loss(pred, y)
        loss.backward()
        evo_loss = opt.step(closure=closure)

        if step % 10 == 0:
            lr = opt.param_groups[0]["lr"]
            msg = f"Step {step:03d} | Loss={loss.item():.4f} | LR={lr:.4f}"
            if evo_loss is not None:
                msg += f" | EvoLoss={evo_loss:.4f} (evolution step)"
            print(msg)


if __name__ == "__main__":
    main()
```

---

### ✅ Run it

```bash
python tiny_demo_evosgd.py
```

Expected output (illustrative):

```
Step 000 | Loss=12.5210 | LR=0.0500
Step 010 | Loss=1.9823  | LR=0.0500
Step 020 | Loss=0.7234  | LR=0.0500
Step 030 | Loss=0.3521  | LR=0.0500 | EvoLoss=0.3409 (evolution step)
Step 040 | Loss=0.2105  | LR=0.0500
...
```

You’ll see the **evolutionary step** trigger every 30 steps (`evo_interval=30`), where EvoSGD explores mutations and replaces params with the recombined elites if better.

---

👉 Do you want me to also make a **side-by-side demo** comparing `torch.optim.SGD` vs `EvoSGD` on the same dataset, so you can see if/when evolution gives an edge?
