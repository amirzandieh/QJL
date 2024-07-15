import math
from scipy.linalg import hadamard
import torch

from qjl_kernel import qjl_kernel

class QJLSketch(torch.nn.Module):
    def __init__(self, dim, dim_outlier, device=None, rng=None, rot=True, rht=False):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert len(dim) == 2, "dim should be a tuple of 2 elements"
        self.dim = dim
        self.dim_outlier = dim_outlier
        
        self.proj_dir = self._init_proj_dir(rng).contiguous()
        self.proj_dir_score = self.init_rot_dir().contiguous() if rot else self.proj_dir
        self.proj_dir_score = self.compose_rand_hadamard_transform().contiguous() if rht else self.proj_dir_score
        self.proj_dir_quant = self.proj_dir_score.transpose(0, 1).contiguous()

    def _init_proj_dir(self, rng):
        return torch.randn(self.dim, generator=rng, dtype=torch.float32, device=self.device)

    def init_rot_dir(self):
        rot_matrices = []
        num_chunks = (self.dim[1] + self.dim[0] - 1) // self.dim[0]
        for i in range(num_chunks):
            start_idx = i * self.dim[0]
            end_idx = (i + 1) * self.dim[0]
            q, _ = torch.linalg.qr(self.proj_dir[:, start_idx:end_idx], mode='reduced')
            rot_matrices.append(q)
        return torch.cat(rot_matrices, dim=-1) * math.sqrt(self.dim[0])

    def compose_rand_hadamard_transform(self):
        H = torch.from_numpy(hadamard(self.dim[0], dtype=float) / math.sqrt(self.dim[0])).to(self.device)
        HD = (H * (2. * torch.randint(0, 2, (self.dim[0],), device=self.device) - 1.)).to(self.proj_dir_score.dtype)
        return torch.einsum('dn,dm-> mn', self.proj_dir_score, HD)
        
    def qjl_qunatize(self, data, outlier_mask, proj_dir_quant):
        s = proj_dir_quant.shape[0]

        key_states_outlier = data * outlier_mask.unsqueeze(-2)
        key_states_inlier = data * (1 - outlier_mask.unsqueeze(-2))
        
        sketched_key_outlier = torch.einsum('...nd,...sd -> ...ns', key_states_outlier.to(proj_dir_quant.dtype), proj_dir_quant)
        sketched_key_inlier = torch.einsum('...nd,...sd -> ...ns', key_states_inlier.to(proj_dir_quant.dtype), proj_dir_quant)
        
        bit_pack_len = 8
        sketched_key_outlier = sketched_key_outlier.view(*sketched_key_outlier.shape[:-1], -1, bit_pack_len)
        sketched_key_inlier = sketched_key_inlier.view(*sketched_key_inlier.shape[:-1], -1, bit_pack_len)
        
        mask_outlier = sketched_key_outlier > 0
        mask_inlier = sketched_key_inlier > 0
        
        enc_vec = 2 ** torch.arange(bit_pack_len, dtype=torch.uint8, device='cuda').view(1, 1, 1, -1)
        
        hash_key_outlier_simhash = (mask_outlier * enc_vec).sum(dim=-1, dtype=torch.uint8)
        hash_key_inlier_simhash = (mask_inlier * enc_vec).sum(dim=-1, dtype=torch.uint8)
        
        hash_key_outlier_simhash = hash_key_outlier_simhash[:,:,:,:,:s//16]

        return hash_key_inlier_simhash, hash_key_outlier_simhash

    def quantize(self, data, outlier_indices):
        assert data.shape[-1] == self.dim[0], 'embedding dimension should match projection dimension'
        assert data.shape[:3] == outlier_indices.shape[:3], 'outlier indices shape should match input shape'
        key_quant, key_outliers_quant, key_outliers_norm = qjl_kernel.qjl_quant(data.contiguous(), outlier_indices.contiguous(), self.proj_dir_quant, self.dim_outlier)
        return key_quant, key_outliers_quant, key_outliers_norm

    def calc_score(self, query, data_quant, outlier_quant, outlier_indices, norm_data, norm_outlier):
        assert query.shape[-1] == self.dim[0], 'embedding dimension should match projection dimension'
        assert query.shape[:2] == data_quant.shape[:2] == outlier_indices.shape[:2], 'query shape should match outlier indices and data quant shapes'
        assert data_quant.shape[:4] == norm_data.shape[:4], 'data quant and its norm should have same shape'
        assert outlier_quant.shape[:4] == norm_outlier.shape[:4], 'outlier quant and its norm should have same shape'
        sketched_q = torch.matmul(query.to(self.proj_dir_score.dtype), self.proj_dir_score)
        scores = qjl_kernel.qjl_score(data_quant.contiguous(),
                                      outlier_quant.contiguous(),
                                      norm_data.contiguous(),
                                      norm_outlier.contiguous(),
                                      outlier_indices.contiguous(),
                                      sketched_q.contiguous(),
                                      query.contiguous(),
                                      self.proj_dir_score,)

        return scores


class QJLKeyQuantizer:
    def __init__(self, qjl_sketch: QJLSketch, outliers_count: int, buffer_size: int, group_size: int, qjl_dim: int) -> None:
        self.qjl_sketch = qjl_sketch
        self.outliers_count = outliers_count
        self.buffer_size = buffer_size
        self.group_size = group_size
        self.qjl_dim = qjl_dim
        self.seq_len = None
        self.outlier_indices = None
        self.key_states_outliers = None
        self.key_states_quant_binary = None
        self.key_states_norm = None
        self.key_residual = None
        self.bit_pack_len = 8

    def build_sketch(self, key_states: torch.Tensor) -> None:
        b, h, _, dim = key_states.shape
        self.seq_len = key_states.shape[-2]
        residual_size = self.seq_len % self.buffer_size

        if residual_size > 0:
            self.key_residual = key_states[:, :, self.seq_len-residual_size:, :]
        if residual_size == self.seq_len:
            return None

        num_groups = (self.seq_len - residual_size) // self.group_size
        key_states = key_states[:, :, :self.seq_len-residual_size, :].reshape((b, h, num_groups, self.group_size, dim)).contiguous()
        
        norms = key_states.norm(dim=-2)

        _, outlier_indices = norms.topk(self.outliers_count, dim=-1)
        self.outlier_indices = outlier_indices.to(torch.uint8).contiguous()

        self.key_states_quant, self.key_outliers_quant, self.key_outliers_norm = self.qjl_sketch.quantize(key_states, self.outlier_indices)

        self.key_states_norm = torch.norm(key_states, dim=-1)

    def _update_norms(self) -> None:
        residual_norm = torch.norm(self.key_residual, dim=-1)
        self.key_states_norm = torch.cat([self.key_states_norm, residual_norm], dim=2).contiguous()

    def _update_qjl(self, outlier_indices) -> None:
        key_states_quant, key_outliers_quant, residual_outliers_norm = self.qjl_sketch.quantize(self.key_residual, outlier_indices)

        self.key_states_quant = torch.cat([self.key_states_quant, key_states_quant], dim=2).contiguous()
        self.key_outliers_quant = torch.cat([self.key_outliers_quant, key_outliers_quant], dim=2).contiguous()
        self.key_outliers_norm = torch.cat([self.key_outliers_norm, residual_outliers_norm], dim=2).contiguous()

    def _update_outliers(self) -> torch.Tensor:
        b, h, num_groups, _, dim = self.key_residual.shape
        norms = self.key_residual.norm(dim=-2)
        _, outlier_indices = norms.topk(self.outliers_count, dim=-1)
        outlier_indices = outlier_indices.to(torch.uint8) 
        self.outlier_indices = torch.cat([self.outlier_indices, outlier_indices], dim=2).contiguous()
        return outlier_indices

    def update_sketch(self, key_states: torch.Tensor) -> None:
        assert key_states.shape[-2] == 1, 'appending more than one embedding in the stream!'
        self.seq_len += 1

        if self.key_residual != None:
            self.key_residual = torch.cat([self.key_residual, key_states], dim=-2)
        else:
            self.key_residual = key_states

        if self.seq_len % self.buffer_size !=0:
            return None

        b, h, _, dim = self.key_residual.shape
        self.key_residual = self.key_residual.reshape((b, h, -1, self.group_size, dim))

        outlier_indices = self._update_outliers()
        self._update_qjl(outlier_indices)
        self._update_norms()

        self.key_residual = None

    def attention_score(self, query_states: torch.Tensor) -> torch.Tensor:
        b, h, _, dim = query_states.shape
        assert query_states.shape[-2] == 1, 'appending more than one embedding in the stream!'
        residual = None
        if self.key_residual != None:
            residual = torch.matmul(query_states, self.key_residual.transpose(-1, -2))

        scores = self.qjl_sketch.calc_score(query_states,
                                            self.key_states_quant,
                                            self.key_outliers_quant,
                                            self.outlier_indices,
                                            self.key_states_norm,
                                            self.key_outliers_norm,
                                            ).transpose(-1, -2)

        if residual != None:
            return torch.cat([scores, residual], dim=-1)
        return scores
