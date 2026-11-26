import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models import build_cnn, build_vgg, unfreeze_block5


def test_cnn_params():
    assert build_cnn().count_params() == 27809
    assert build_cnn(batchnorm=True).count_params() == 28257


def test_vgg_params_and_unfreeze():
    m, base = build_vgg(pretrained=False)
    assert m.count_params() == 14715201 and sum(int(np.prod(w.shape)) for w in m.trainable_weights) == 513
    unfreeze_block5(m, base)
    assert sum(int(np.prod(w.shape)) for w in m.trainable_weights) > 7_000_000
