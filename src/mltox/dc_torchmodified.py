from deepchem.models.torch_models.torch_model import TorchModel
from typing import Any, Iterable, List, Optional, Tuple
from deepchem.trans import Transformer, undo_transforms
from deepchem.utils.typing import OneOrMany
import numpy as np
import torch

class ModifiedTorchModel(TorchModel):

   def __init__(self,*args,**kwargs):
        super(ModifiedTorchModel, self).__init__(*args,**kwargs)


  
   def _predict(self, generator: Iterable[Tuple[Any, Any, Any]],
               transformers: List[Transformer], uncertainty: bool,
               other_output_types: Optional[OneOrMany[str]]):

    results: Optional[List[List[np.ndarray]]] = None
    variances: Optional[List[List[np.ndarray]]] = None
    if uncertainty and (other_output_types is not None):
      raise ValueError(
          'This model cannot compute uncertainties and other output types simultaneously. Please invoke one at a time.'
      )
    if uncertainty:
      if self._variance_outputs is None or len(self._variance_outputs) == 0:
        raise ValueError('This model cannot compute uncertainties')
      if len(self._variance_outputs) != len(self._prediction_outputs):
        raise ValueError(
            'The number of variances must exactly match the number of outputs')
    if other_output_types:
      if self._other_outputs is None or len(self._other_outputs) == 0:
        raise ValueError(
            'This model cannot compute other outputs since no other output_types were specified.'
        )
    self._ensure_built()
    self.model.eval()
    for batch in generator:
      inputs, labels, weights = batch
      inputs, _, _ = self._prepare_batch((inputs, None, None))

      # Invoke the model.
      if isinstance(inputs, list) and len(inputs) == 1:
        inputs = inputs[0]
      output_values = self.model(inputs)
      if isinstance(output_values, torch.Tensor):
        output_values = [output_values]
      if len(output_values) > 3:
        output_values = output_values[:3]
      output_values = [t.detach().cpu().numpy() for t in output_values]

      # Apply tranformers and record results.
      if uncertainty:
        var = [output_values[i] for i in self._variance_outputs]
        if variances is None:
          variances = [var]
        else:
          for i, t in enumerate(var):
            variances[i].append(t)
      access_values = []
      if other_output_types:
        access_values += self._other_outputs
      elif self._prediction_outputs is not None:
        access_values += self._prediction_outputs


      if len(access_values) > 0:
        output_values = [output_values[i] for i in access_values]

      if len(transformers) > 0:
        if len(output_values) > 1:
          raise ValueError(
              "predict() does not support Transformers for models with multiple outputs."
          )
        elif len(output_values) == 1:
          output_values = [undo_transforms(output_values[0], transformers)]
      if results is None:
        results = [[] for i in range(len(output_values))]
      for i, t in enumerate(output_values):
        results[i].append(t)

    # Concatenate arrays to create the final results.
    final_results = []
    final_variances = []
    if results is not None:
      for r in results:
        final_results.append(np.concatenate(r, axis=0))
    if uncertainty and variances is not None:
      for v in variances:
        final_variances.append(np.concatenate(v, axis=0))
      return zip(final_results, final_variances)
    if len(final_results) == 1:
      return final_results[0]
    else:
      return final_results