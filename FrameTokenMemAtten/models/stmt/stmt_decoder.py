from models.skeleton.skeleton_decoder import SkeletonDecodeModel


class StatementDecodeModel(SkeletonDecodeModel):
  
  def __init__(self):
    super(StatementDecodeModel, self).__init__()
    self.treat_first_element_as_skeleton = 0
  

