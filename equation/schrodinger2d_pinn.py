import torch 

def realpart_2d(x, y, kx=8, ky=8, px=2.0, py=0.0):
    x01 = (x + 1) / 2
    y01 = (y + 1) / 2
    phase = px * (x01 - 0.5) + py * (y01 - 0.5)
    envelope = torch.exp(-0.5 * (kx * (x01 - 0.5)**2 + ky * (y01 - 0.5)**2))
    return envelope * torch.cos(phase)

def imagpart_2d(x, y, kx=8, ky=8, px=2.0, py=0.0):
    x01 = (x + 1) / 2
    y01 = (y + 1) / 2
    phase = px * (x01 - 0.5) + py * (y01 - 0.5)
    envelope = torch.exp(-0.5 * (kx * (x01 - 0.5)**2 + ky * (y01 - 0.5)**2))
    return envelope * torch.sin(phase)

class Schrodinger_2D:
    def __init__(self, x0, y0 , kx, ky):
        self.x0 = x0
        self.y0 = y0
        self.kx = kx
        self.ky = ky

    @staticmethod
    def normalize(x):
        return 2 * x - 1

    def potential(self, x, y):
        x = (x + 1) / 2  
        y = (y + 1) / 2
        return 0.5 * (x**2 * self.kx**2 + y**2 * self.ky**2)

    def sample_points(self, N_i=8000, N_p=8000):
        x_i = torch.rand(N_i, 1)
        y_i = torch.rand(N_i, 1)
        t_i = torch.zeros_like(x_i)

        x_p = torch.rand(N_p, 1)
        y_p = torch.rand(N_p, 1)
        t_p = torch.rand(N_p, 1)

        x_i = self.normalize(x_i)
        y_i = self.normalize(y_i)
        x_p = self.normalize(x_p)
        y_p = self.normalize(y_p)
        t_p = self.normalize(t_p)

        return x_i, y_i, t_i, x_p, y_p, t_p

    def physics_loss(self, model, x, y, t, norm_weight=15, resolution=100):
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)

        X = torch.cat([x, y, t], dim=-1)
        u, v = model(X)

        ut = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        ux = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        uy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        vt = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        vx = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        vy = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        uxx = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=True)[0]
        uyy = torch.autograd.grad(uy, y, grad_outputs=torch.ones_like(uy), create_graph=True)[0]
        vxx = torch.autograd.grad(vx, x, grad_outputs=torch.ones_like(vx), create_graph=True)[0]
        vyy = torch.autograd.grad(vy, y, grad_outputs=torch.ones_like(vy), create_graph=True)[0]

        V = self.potential(x, y)
        u_residual = ut + (0.5 * (vxx + vyy)) - (V * v)
        v_residual = vt - (0.5 * (uxx + uyy)) + (V * u)
        pde_loss = torch.mean(u_residual**2 + v_residual**2)

        x_grid = torch.linspace(0, 1, resolution)
        y_grid = torch.linspace(0, 1, resolution)
        dx = 1.0 / resolution
        dy = 1.0 / resolution
        norm_loss_total = 0.0

        for t_val in torch.linspace(0, 1, 5):
            x_mesh, y_mesh = torch.meshgrid(x_grid, y_grid, indexing="ij")
            x_flat = x_mesh.reshape(-1, 1)
            y_flat = y_mesh.reshape(-1, 1)
            t_fixed = torch.full_like(x_flat, t_val)
            X_input = torch.cat([self.normalize(x_flat), self.normalize(y_flat), self.normalize(t_fixed)], dim=1)
            u_norm, v_norm = model(X_input)
            prob = u_norm**2 + v_norm**2
            norm_val = torch.sum(prob) * dx * dy
            penalty = ((norm_val - 1) / (norm_val + 1e-8))**2
            norm_loss_total += penalty

        norm_loss = norm_loss_total / 5
        return pde_loss + norm_weight * norm_loss

    def initial_loss(self, model, x, y, t):
        mask = (t == 0).squeeze()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=x.device)
        x0 = x[mask]
        y0 = y[mask]
        t0 = t[mask]
        X = torch.cat([x0, y0, t0], dim=1)
        u, v = model(X)

        with torch.no_grad():
            u0 = realpart_2d(x0, y0, self.kx, self.ky)
            v0 = imagpart_2d(x0, y0)
            prob_density = u0**2 + v0**2
            norm0 = torch.mean(prob_density)

        u0 = u0 / torch.sqrt(norm0)
        v0 = v0 / torch.sqrt(norm0)

        eps = 1e-8
        loss_u = torch.sum(torch.abs(u - u0)) / (u.numel() + eps)
        loss_v = torch.sum(torch.abs(v - v0)) / (v.numel() + eps)
        return loss_u + loss_v

    def boundary_loss(self, model, N_b=4000):
        x = torch.rand(N_b, 1)
        y = torch.rand(N_b, 1)
        t = torch.rand(N_b, 1)

        X_left = torch.cat([self.normalize(torch.zeros_like(x)), self.normalize(y), self.normalize(t)], dim=1)
        X_right = torch.cat([self.normalize(torch.ones_like(x)), self.normalize(y), self.normalize(t)], dim=1)
        X_top = torch.cat([self.normalize(x), self.normalize(torch.ones_like(y)), self.normalize(t)], dim=1)
        X_bottom = torch.cat([self.normalize(x), self.normalize(torch.zeros_like(y)), self.normalize(t)], dim=1)

        u_l, v_l = model(X_left)
        u_r, v_r = model(X_right)
        u_t, v_t = model(X_top)
        u_b, v_b = model(X_bottom)

        eps = 1e-8
        bound_loss = 0.25 * (
            torch.sum(torch.abs(u_l)) / (u_l.numel() + eps) +
            torch.sum(torch.abs(v_l)) / (v_l.numel() + eps) +
            torch.sum(torch.abs(u_r)) / (u_r.numel() + eps) +
            torch.sum(torch.abs(v_r)) / (v_r.numel() + eps) +
            torch.sum(torch.abs(u_t)) / (u_t.numel() + eps) +
            torch.sum(torch.abs(v_t)) / (v_t.numel() + eps) +
            torch.sum(torch.abs(u_b)) / (u_b.numel() + eps) +
            torch.sum(torch.abs(v_b)) / (v_b.numel() + eps)
        )
        return bound_loss

    def compute_norm(self, model, t_fixed=0.5, resolution=100):
        x = torch.linspace(0, 1, resolution)
        y = torch.linspace(0, 1, resolution)
        x_mesh, y_mesh = torch.meshgrid(x, y, indexing="ij")
        x_flat = x_mesh.reshape(-1, 1)
        y_flat = y_mesh.reshape(-1, 1)
        t_fixed = torch.full_like(x_flat, t_fixed)

        X_input = torch.cat([
            self.normalize(x_flat),
            self.normalize(y_flat),
            self.normalize(t_fixed)
        ], dim=1)

        with torch.no_grad():
            u, v = model(X_input)
            prob_density = u**2 + v**2
            dx = 1.0 / resolution
            dy = 1.0 / resolution
            return torch.sum(prob_density) * dx * dy

    def compute_loss(self, model, w_init=10, w_phys=2, w_bound=2):
        x_i, y_i, t_i, x_p, y_p, t_p = self.sample_points()
        L_init = self.initial_loss(model, x_i, y_i, t_i)
        L_phys = self.physics_loss(model, x_p, y_p, t_p)
        L_bound = self.boundary_loss(model)
        total = w_init * L_init + w_phys * L_phys + w_bound * L_bound
        return total, (L_init, L_phys, L_bound)
