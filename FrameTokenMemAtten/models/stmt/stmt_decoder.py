from models.skeleton.skeleton_decoder import SkeletonDecodeModel


class StatementDecodeModel(SkeletonDecodeModel):
  
  def __init__(self, type_content_data):
    super(StatementDecodeModel, self).__init__(type_content_data)
    self.treat_first_element_as_skeleton = 0
  

