import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_sizes = [config["input"]] + config["hiddens"]
        out_sizes = config["hiddens"] + [config["output"]]
        layers = []
        for in_size, out_size, use_bias, dropout, use_bn, use_ln, activation in zip(
                in_sizes,
                out_sizes,
                config["use_biases"],
                config["dropouts"],
                config["use_batch_norms"],
                config["use_layer_norms"],
                config["activations"],
        ):
            layers.append(nn.Linear(in_size, out_size, use_bias))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            if use_bn:
                layers.append(nn.BatchNorm1d(out_size))
            if use_ln:
                layers.append(nn.LayerNorm(out_size))
            if activation is not None:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "sigmoid":
                    layers.append(nn.Sigmoid())
                elif activation == "softmax":
                    layers.append(nn.Softmax())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation.startswith("leaky_relu"):
                    neg_slope = float(activation.split(":")[1])
                    layers.append(nn.LeakyReLU(neg_slope))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LinBnDrop(nn.Sequential):
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=True):
        layers = [nn.BatchNorm1d(n_out if lin_first else n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin + layers if lin_first else layers + lin
        super().__init__(*layers)

class Encoder(nn.Module):
    def __init__(self, args, nfeatures_modality1=1, nfeatures_modality2=1, hidden_dim=150, z_dim=64):
        super(Encoder, self).__init__()
        self.nfeatures_modality1 = nfeatures_modality1
        self.nfeatures_modality2 = nfeatures_modality2
        self.hidden_dim = hidden_dim

        self.encoder_modality1 = LinBnDrop(self.nfeatures_modality1, self.hidden_dim, p=0.2, act=nn.ReLU())

        if self.nfeatures_modality2 < self.hidden_dim:
            self.encoder_modality2 = LinBnDrop(self.nfeatures_modality2, self.hidden_dim, p=0.1, act=nn.ReLU())
        else:
            self.encoder_modality2 = LinBnDrop(self.nfeatures_modality2, self.hidden_dim, p=0.2, act=nn.ReLU())
        self.weights_modality1 = nn.Parameter(torch.rand((1, self.nfeatures_modality1)) * 0.001, requires_grad=True)
        self.weights_modality2 = nn.Parameter(torch.rand((1, self.nfeatures_modality2)) * 0.001, requires_grad=True)

        self.spec_extract1 = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )
        self.spec_extract2 = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )
        self.query_initial_shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

        self.gate_s1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.gate_s2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.gate_sh = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.ln_shared = nn.LayerNorm(self.hidden_dim)
        self.ln_spec1 = nn.LayerNorm(self.hidden_dim)
        self.ln_spec2 = nn.LayerNorm(self.hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.w_q = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.w_k = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.w_v = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.alpha = nn.Parameter(torch.tensor(0.2))
        self.q_norm = nn.LayerNorm(self.hidden_dim)
        self.k_norm = nn.LayerNorm(self.hidden_dim)

        self.attn_gate = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.feature_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid()
        )

        args.encoder['input'] = 150
        self.encoder_vae_path = MLP(args.encoder)
        self.fc_mu = LinBnDrop(z_dim, z_dim, p=0.2)
        self.fc_var = LinBnDrop(z_dim, z_dim, p=0)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        h1 = self.encoder_modality1(x[:, :self.nfeatures_modality1]* self.weights_modality1)
        h2 = self.encoder_modality2(x[:, self.nfeatures_modality1:]* self.weights_modality2)

        gate_init = torch.softmax(torch.cat([self.query_initial_shared(h1), self.query_initial_shared(h2)], dim=-1),
                                  dim=-1)
        shared_init = gate_init[:, 0].unsqueeze(1) * h1 + gate_init[:, 1].unsqueeze(1) * h2


        spec1 = self.ln_spec1(self.spec_extract1(torch.cat([h1, shared_init], dim=-1)))
        spec2 = self.ln_spec2(self.spec_extract2(torch.cat([h2, shared_init], dim=-1)))
        shared = self.ln_shared(shared_init)


        # 门控调节
        spec1 = spec1 * self.gate_s1(spec1)
        spec2 = spec2 * self.gate_s2(spec2)
        shared = shared * self.gate_sh(shared)

        Q_unnorm = self.w_q(shared)
        Q = self.q_norm(Q_unnorm).unsqueeze(1)
        KV_seq = torch.stack([spec1, spec2], dim=1)
        K_unnorm = self.w_k(KV_seq)
        K = self.k_norm(K_unnorm)
        V = self.w_v(KV_seq)
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V).squeeze(1)

        dimension_gate = torch.sigmoid(self.attn_gate(shared))
        filtered_attn_out = attn_out * dimension_gate

        fused_all = shared + self.alpha * filtered_attn_out
        x_vae = self.encoder_vae_path(fused_all)
        mu_unnorm = self.fc_mu(x_vae)
        z=mu_unnorm

        return z, spec1, spec2, shared, h1, h2,mu_unnorm
class Decoder(nn.Module):
    def __init__(self, nfeatures_modality1=1, nfeatures_modality2=1, hidden_dim=150, z_dim=64):
        super(Decoder, self).__init__()
        self.decoder_mod1 = nn.Sequential(
            nn.Linear(z_dim + hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, nfeatures_modality1)
        )
        self.decoder_mod2 = nn.Sequential(
            nn.Linear(z_dim + hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, nfeatures_modality2)
        )

    def forward(self, z, spec1, spec2):
        recon1 = self.decoder_mod1(torch.cat([z, spec1], dim=-1))
        recon2 = self.decoder_mod2(torch.cat([z, spec2], dim=-1))
        return torch.cat([recon1, recon2], dim=-1)

class scMoDCA(nn.Module):
    def __init__(self, args, nfeatures_modality1=1, nfeatures_modality2=1, hidden_dim=150, z_dim=64, classify_dim=1):
        super(scMoDCA, self).__init__()
        self.nfeatures_modality1 = nfeatures_modality1
        self.nfeatures_modality2 = nfeatures_modality2
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.classify_dim = classify_dim
        self.encoder = Encoder(args, nfeatures_modality1, nfeatures_modality2, hidden_dim, z_dim)
        self.classify = nn.Linear(z_dim, classify_dim)
        self.decoder = Decoder(nfeatures_modality1, nfeatures_modality2, hidden_dim, z_dim)
        critic_hidden = 1
        self.club1 = CLUBInfoNCECritic(hidden_dim, hidden_dim, critic_hidden, 1, 'relu')
        self.club2 = CLUBInfoNCECritic(hidden_dim, hidden_dim, critic_hidden, 1, 'relu')
        self.infonce_mod1_shared = InfoNCECritic(self.hidden_dim, self.hidden_dim, 1, 1, 'relu')
        self.infonce_mod2_shared = InfoNCECritic(self.hidden_dim, self.hidden_dim,1, 1, 'relu')
        self.club_cond_1 = CLUBInfoNCECritic(self.hidden_dim * 2, self.hidden_dim, critic_hidden, 1, 'relu')
        self.club_cond_2 = CLUBInfoNCECritic(self.hidden_dim * 2, self.hidden_dim, critic_hidden, 1, 'relu')

    def forward(self, x):
        z, spec1, spec2, shared, orig_m1, orig_m2,mu_unnorm = self.encoder(x)
        x_res = self.decoder(z, spec1, spec2)
        x_cluster = self.classify(z)


        return x_res, z, spec1, spec2, shared,orig_m1, orig_m2, mu_unnorm,x_cluster




def build_mlp(dim, hidden_dim, output_dim, layers, activation):
    activation = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
    }[activation]

    seq = [nn.Linear(dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)

class InfoNCECritic(nn.Module):
    def __init__(self, A_dim, B_dim, hidden_dim, layers, activation,  **extra_kwargs):
        super(InfoNCECritic, self).__init__()
        self.temperature = 0.5
        self._f = build_mlp(A_dim + B_dim, hidden_dim, 1, layers, activation)

    def forward(self, x_samples, y_samples):
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self._f(torch.cat([x_samples, y_samples], dim=-1)) / self.temperature
        T1 = self._f(torch.cat([x_tile, y_tile], dim=-1)) / self.temperature

        lower_bound = T0.mean() - (T1.logsumexp(dim=1).mean() - np.log(sample_size))

        return -lower_bound * self.temperature


class CLUBInfoNCECritic(nn.Module):
    def __init__(self, A_dim, B_dim, hidden_dim, layers, activation, **extra_kwargs):
        super(CLUBInfoNCECritic, self).__init__()

        self._f = build_mlp(A_dim + B_dim, hidden_dim, 1, layers, activation)


    # CLUB loss
    def forward(self, x_samples, y_samples):
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self._f(torch.cat([y_samples, x_samples], dim=-1))
        T1 = self._f(torch.cat([y_tile, x_tile], dim=-1))

        return T0.mean() - T1.mean()

    def learning_loss(self, x_samples, y_samples):
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self._f(torch.cat([y_samples, x_samples], dim=-1))
        T1 = self._f(torch.cat([y_tile, x_tile], dim=-1))

        lower_bound = T0.mean() - (T1.logsumexp(dim=1).mean() - np.log(sample_size))
        return -lower_bound

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss




