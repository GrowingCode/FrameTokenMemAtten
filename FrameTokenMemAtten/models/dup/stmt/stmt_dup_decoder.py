from models.dup.skeleton.skeleton_dup_decoder import SkeletonDupModel


class StatementDupModel(SkeletonDupModel):
  
  def __init__(self, type_content_data):
    super(StatementDupModel, self).__init__(type_content_data)
    self.treat_first_element_as_skeleton = 0
  
  
  
  
  




