def _normalize_model_name(model):
    if model is None:
        raise ValueError('predictor model name is required')

    model_name = str(model).strip().lower().replace('-', '_').replace(' ', '_')
    aliases = {
        'adaptive_switching': 'as',
        'adaptive_switching_ensemble': 'as',
        'adaptive': 'as',
        'cart': 'carts',
        'carts': 'carts',
        'random_forest': 'carts',
    }
    return aliases.get(model_name, model_name)


def get_acc_predictor(model, inputs, targets):
    model = _normalize_model_name(model)

    if model == 'rbf':
        from acc_predictor.rbf import RBF
        acc_predictor = RBF()
        acc_predictor.fit(inputs, targets)

    elif model == 'carts':
        from acc_predictor.carts import CART
        acc_predictor = CART(n_tree=5000)
        acc_predictor.fit(inputs, targets)

    elif model == 'gp':
        from acc_predictor.gp import GP
        acc_predictor = GP()
        acc_predictor.fit(inputs, targets)

    elif model == 'mlp':
        from acc_predictor.mlp import MLP
        acc_predictor = MLP(n_feature=inputs.shape[1])
        acc_predictor.fit(x=inputs, y=targets)

    elif model == 'as':
        from acc_predictor.adaptive_switching import AdaptiveSwitching
        acc_predictor = AdaptiveSwitching()
        acc_predictor.fit(inputs, targets)

    else:
        raise NotImplementedError

    return acc_predictor


def get_acc_predictor_from_config(config, inputs, targets):
    if isinstance(config, str):
        model = config
    elif isinstance(config, dict):
        search_cfg = config.get('search', config)
        model = search_cfg.get('predictor', config.get('predictor'))
    else:
        model = getattr(config, 'predictor', None)

    return get_acc_predictor(model, inputs, targets)

