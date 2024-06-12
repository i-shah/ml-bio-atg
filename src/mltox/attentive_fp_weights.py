import torch.nn.functional as F
import torch.nn as nn

import torch

from .dc_torchmodified import ModifiedTorchModel
from deepchem.models.torch_models.torch_model import TorchModel

class AttentiveFPWeights(nn.Module):
  def __init__(self,
               n_tasks: int,
               num_layers: int = 2,
               num_timesteps: int = 2,
               graph_feat_size: int = 200,
               dropout: float = 0.,
               mode: str = 'regression',
               number_atom_features: int = 30,
               number_bond_features: int = 11,
               n_classes: int = 2,
               nfeat_name: str = 'x',
               efeat_name: str = 'edge_attr'):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks.
    num_layers: int
      Number of graph neural network layers, i.e. number of rounds of message passing.
      Default to 2.
    num_timesteps: int
      Number of time steps for updating graph representations with a GRU. Default to 2.
    graph_feat_size: int
      Size for graph representations. Default to 200.
    dropout: float
      Dropout probability. Default to 0.
    mode: str
      The model type, 'classification' or 'regression'. Default to 'regression'.
    number_atom_features: int
      The length of the initial atom feature vectors. Default to 30.
    number_bond_features: int
      The length of the initial bond feature vectors. Default to 11.
    n_classes: int
      The number of classes to predict per task
      (only used when ``mode`` is 'classification'). Default to 2.
    nfeat_name: str
      For an input graph ``g``, the model assumes that it stores node features in
      ``g.ndata[nfeat_name]`` and will retrieve input node features from that.
      Default to 'x'.
    efeat_name: str
      For an input graph ``g``, the model assumes that it stores edge features in
      ``g.edata[efeat_name]`` and will retrieve input edge features from that.
      Default to 'edge_attr'.
    """
    try:
      import dgl
    except:
      raise ImportError('This class requires dgl.')
    try:
      import dgllife
    except:
      raise ImportError('This class requires dgllife.')

    if mode not in ['classification', 'regression']:
      raise ValueError("mode must be either 'classification' or 'regression'")

    super(AttentiveFPWeights, self).__init__()

    self.n_tasks = n_tasks
    self.mode = mode
    self.n_classes = n_classes
    self.nfeat_name = nfeat_name
    self.efeat_name = efeat_name
    if mode == 'classification':
      out_size = n_tasks * n_classes
    else:
      out_size = n_tasks

    from dgllife.model import AttentiveFPPredictor as DGLAttentiveFPPredictor

    self.model = DGLAttentiveFPPredictor(
        node_feat_size=number_atom_features,
        edge_feat_size=number_bond_features,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        graph_feat_size=graph_feat_size,
        n_tasks=out_size,
        dropout=dropout)

  def forward(self, g):
    """Predict graph labels
    Parameters
    ----------
    g: DGLGraph
      A DGLGraph for a batch of graphs. It stores the node features in
      ``dgl_graph.ndata[self.nfeat_name]`` and edge features in
      ``dgl_graph.edata[self.efeat_name]``.
    Returns
    -------
    torch.Tensor
      The model output.
      * When self.mode = 'regression',
        its shape will be ``(dgl_graph.batch_size, self.n_tasks)``.
      * When self.mode = 'classification', the output consists of probabilities
        for classes. Its shape will be
        ``(dgl_graph.batch_size, self.n_tasks, self.n_classes)`` if self.n_tasks > 1;
        its shape will be ``(dgl_graph.batch_size, self.n_classes)`` if self.n_tasks is 1.
    torch.Tensor, optional
      This is only returned when self.mode = 'classification', the output consists of the
      logits for classes before softmax.
    """
    node_feats = g.ndata[self.nfeat_name]
    edge_feats = g.edata[self.efeat_name]
    out, node_weights = self.model(g, node_feats, edge_feats, get_node_weight=True)

    if self.mode == 'classification':
      if self.n_tasks == 1:
        logits = out.view(-1, self.n_classes)
        softmax_dim = 1
      else:
        logits = out.view(-1, self.n_tasks, self.n_classes)
        softmax_dim = 2
      proba = F.softmax(logits, dim=softmax_dim)
      return proba, logits, out, node_weights
    else:
      return out

from deepchem.models.losses import Loss, L2Loss, SparseSoftmaxCrossEntropy
from deepchem.models.torch_models.torch_model import TorchModel

class AttentiveFPModelWeights(ModifiedTorchModel):
  def __init__(self,
               n_tasks: int,
               num_layers: int = 2,
               num_timesteps: int = 2,
               graph_feat_size: int = 200,
               dropout: float = 0.,
               mode: str = 'regression',
               number_atom_features: int = 30,
               number_bond_features: int = 11,
               n_classes: int = 2,
               self_loop: bool = True,
               **kwargs):
               
    model = AttentiveFPWeights(
        n_tasks=n_tasks,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        graph_feat_size=graph_feat_size,
        dropout=dropout,
        mode=mode,
        number_atom_features=number_atom_features,
        number_bond_features=number_bond_features,
        n_classes=n_classes)

    if mode == 'regression':
      loss: Loss = L2Loss()
      output_types = ['prediction']
    else:
      loss = SparseSoftmaxCrossEntropy()
      output_types = ['prediction', 'loss','embedding']
    super(AttentiveFPModelWeights, self).__init__(
        model, loss=loss, output_types=output_types, **kwargs)

    self._self_loop = self_loop

  def _prepare_batch(self, batch):
    """Create batch data for AttentiveFP.
    Parameters
    ----------
    batch: tuple
      The tuple is ``(inputs, labels, weights)``.
    Returns
    -------
    inputs: DGLGraph
      DGLGraph for a batch of graphs.
    labels: list of torch.Tensor or None
      The graph labels.
    weights: list of torch.Tensor or None
      The weights for each sample or sample/task pair converted to torch.Tensor.
    """
    try:
      import dgl
    except:
      raise ImportError('This class requires dgl.')

    inputs, labels, weights = batch
    dgl_graphs = [
        graph.to_dgl_graph(self_loop=self._self_loop) for graph in inputs[0]
    ]

    inputs = dgl.batch(dgl_graphs).to(self.device)
    _, labels, weights = super(AttentiveFPModelWeights, self)._prepare_batch(
        ([], labels, weights))
    return inputs, labels, weights