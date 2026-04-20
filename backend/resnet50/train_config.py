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

    def __repr__(self):
        return (
            f"TrainResNetConfig("
            f"unfreeze_layers={self.unfreeze_layers}, "
            f"dropout={self.dropout}, "
            f"threshold={self.threshold}, "
            f"scheduler={self.use_scheduler}, "
            f"model_name='{self.model_name}')"
        )