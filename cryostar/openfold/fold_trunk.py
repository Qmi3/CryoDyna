import torch
import os
import torch.nn as nn
from cryostar.openfold.model.structure_module import StructureModule
from cryostar.openfold.model.heads import AuxiliaryHeads
from cryostar.openfold.utils.tensor_utils import tensor_tree_map
from cryostar.openfold.utils.feats import atom14_to_atom37
from cryostar.openfold.utils.loss import find_structural_violations,violation_loss
class AlphaFold_Structure_Module(nn.Module):
    def __init__(self, config,fold_feature):
        super(AlphaFold_Structure_Module, self).__init__()
        self.globals = config.globals
        self.config = config.model
        self.loss = config.loss
        self.fold_feature = {k:torch.from_numpy(v) for k,v in fold_feature.items()}
        self.structure_module = StructureModule(
            is_multimer=self.globals.is_multimer,
            fold_feature = self.fold_feature,
            **self.config["structure_module"],
        )
        self.aux_heads = AuxiliaryHeads(
            self.config["heads"],
        )

    def iteration(self, feats):
        # Primary output dictionary
        outputs = {}

        # This needs to be done manually for DeepSpeed's sake
        # dtype = next(self.parameters()).dtype
        
        # for k in feats:
        #     if feats[k].dtype == torch.float32:
        #         feats[k] = feats[k].to(dtype=dtype)
        fold_feature =  {k:v.to(feats.device) for k,v in self.fold_feature.items()}
        inplace_safe = not (self.training or torch.is_grad_enabled())
        # outputs['single'] = torch.stack(self.fixed_feats["single"] * feats[''])
        # outputs['pair'] = torch.stack(self.fixed_feats["pair"] * feats.shape[0])
        # if self.globals.offload_inference and inplace_safe:
        # Predict 3D structure where we focus most
        # outputs['single'] = feats
        outputs["sm"] = self.structure_module(
            feats,
            # feats["aatype"],
            # mask=feats["seq_mask"].to(dtype=dtype),
            inplace_safe=inplace_safe,
            _offload_inference=self.globals.offload_inference,
        )
        all_atom_positions = []
        for position in outputs["sm"]["positions"]:
            all_atom_positions.append(atom14_to_atom37(
            position,fold_feature
            ))
        outputs["final_atom_positions"] = all_atom_positions[-1]
        outputs["all_atom_positions"] = torch.stack(all_atom_positions)
        # outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]
        
        # import pdb;pdb.set_trace()
        
        outputs["violation"] = find_structural_violations(
                fold_feature,
                outputs["sm"]["positions"][-1],
                **self.loss["violation"],
            )
        
        outputs['violation_loss'] = violation_loss(
                outputs["violation"],
                **{**fold_feature, **self.loss["violation"]})
        
        return outputs
    
    def forward(self, batch):
        outputs = self.iteration(batch)
        outputs.update(self.aux_heads(outputs))
        return outputs
