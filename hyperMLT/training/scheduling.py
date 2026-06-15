from __future__ import annotations


LOSS_TERMS = ("data", "div", "mom", "temp", "srt")


def _default_loss_weights(config) -> dict[str, float]:

    residual_weights = dict(config.physics.residual_weights or {})

    return {
        term: float(residual_weights.get(term, 0.0 if term != "data" else 1.0))
        for term in LOSS_TERMS
    }


def _sorted_stages(config) -> list[dict]:

    scheduling = dict(config.training.scheduling or {})
    stages = list(scheduling.get("stages", []))

    return sorted(stages, key=lambda stage: int(stage.get("start", 0)))


def _stage_target_weights(previous_targets: dict[str, float], stage: dict) -> dict[str, float]:

    targets = dict(previous_targets)
    stage_weights = dict(stage.get("weights", {}))

    for term in LOSS_TERMS:
        if term in stage_weights:
            targets[term] = float(stage_weights[term])

    return targets


def current_loss_weights(config, epoch: int, total_epochs: int) -> dict[str, float]:

    defaults = _default_loss_weights(config)
    stages = _sorted_stages(config)

    if not stages:
        return defaults

    if int(epoch) < int(stages[0].get("start", 0)):
        return defaults

    current_start_weights = dict(defaults)

    for index, stage in enumerate(stages):
        start_epoch = int(stage.get("start", 0))
        next_start_epoch = (
            int(stages[index + 1].get("start", total_epochs))
            if index + 1 < len(stages)
            else int(total_epochs)
        )
        end_epoch = max(start_epoch, next_start_epoch - 1)
        target_weights = _stage_target_weights(current_start_weights, stage)

        if epoch < start_epoch:
            return current_start_weights

        if epoch <= end_epoch:
            if end_epoch <= start_epoch:
                return target_weights

            progress = float(epoch - start_epoch) / float(end_epoch - start_epoch)

            return {
                term: current_start_weights[term]
                + progress * (target_weights[term] - current_start_weights[term])
                for term in LOSS_TERMS
            }

        current_start_weights = target_weights

    return current_start_weights
