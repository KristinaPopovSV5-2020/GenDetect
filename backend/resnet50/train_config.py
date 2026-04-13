class TrainResNetConfig:
    def __init__(
        self,
        unfreeze_layers=("layer4",),   # ("layer3","layer4") or ("layer4",)
        dropout=0.5,
        threshold=0.5,
        use_scheduler=False,
        model_name="resnet_model.pth"
    ):
        self.unfreeze_layers = unfreeze_layers
        self.dropout = dropout
        self.threshold = threshold
        self.use_scheduler = use_scheduler
        self.model_name = model_name