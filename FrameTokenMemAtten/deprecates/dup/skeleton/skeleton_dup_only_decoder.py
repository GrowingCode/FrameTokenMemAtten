from models.skeleton.skeleton_only_decoder import SkeletonOnlyDecodeModel


class SkeletonDupOnlyDecodeModel(SkeletonOnlyDecodeModel):
  
  def __init__(self, type_content_data, compute_noavg = True):
    super(SkeletonDupOnlyDecodeModel, self).__init__(type_content_data, compute_noavg)
    self.skt_seq_acc_control = False
  
  













