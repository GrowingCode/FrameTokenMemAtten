from inputs.example_data_loader import build_statement_dup_feed_dict
from metas.hyper_settings import model_run_mode, statement_dup_mode
from models.dup.skeleton.skeleton_dup_runner import SkeletonDupRunner
from models.dup.stmt.stmt_dup_decoder import StatementDupModel


class StatementDupRunner(SkeletonDupRunner):
  
  def __init__(self, sess):
    super(StatementDupRunner, self).__init__(sess)
    
  def set_up_example_loader(self):
    self.example_loader = build_statement_dup_feed_dict
  
  def build_model_logic(self):
    assert model_run_mode == statement_dup_mode, "serious error! not statement dup mode? but the model logic is statement dup logic."
    self.model = StatementDupModel(self.type_content_data)
  


