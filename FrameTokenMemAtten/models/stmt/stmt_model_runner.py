from inputs.example_data_loader import build_statement_feed_dict
from metas.hyper_settings import model_run_mode, statement_decode_mode
from models.skeleton.skeleton_model_runner import SkeletonModelRunner
from models.stmt.stmt_decoder import StatementDecodeModel


class StatementModelRunner(SkeletonModelRunner):
  
  def __init__(self, sess):
    super(StatementModelRunner, self).__init__(sess)
    
  def set_up_example_loader(self):
    self.example_loader = build_statement_feed_dict
  
  def build_model_logic(self):
    assert model_run_mode == statement_decode_mode, "serious error! not statement decode mode? but the model logic is statement decode logic."
    self.model = StatementDecodeModel(self.type_content_data)
  


