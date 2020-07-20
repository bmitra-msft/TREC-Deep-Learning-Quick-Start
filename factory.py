from model import NDRM1, NDRM2, NDRM3
from loss import SmoothMRRLoss, RankNetLoss, MarginLoss

class Factory:

    models = {'ndrm1': NDRM1, 'ndrm2': NDRM2, 'ndrm3': NDRM3}
    losses = {'smoothmrr': SmoothMRRLoss, 'ranknet': RankNetLoss, 'margin': MarginLoss}

    def get_model(utils):
        return Factory.safe_get('model', utils.args.model, Factory.models)(utils)

    def get_loss(utils, device):
        return Factory.safe_get('loss', utils.args.loss, Factory.losses)().to(device)

    def safe_get(type, name, dict):
        assert name in dict, 'unknown {} {}'.format(type, name)
        return dict[name]
