#!/usr/bin/env python3
"""A/M-only adapter: count consumed weather without changing shared trainer code."""
from collections import Counter
import train_salad
from workflow import training_data


class WeatherCoverage(training_data.SamplingCoverage):
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        super().__init__(dataset, **kwargs)

    def begin_epoch(self):
        super().begin_epoch()
        self.weather = Counter()

    def update(self, metadata):
        super().update(metadata)
        # Stateless replay of CONSUMED places, never prefetched worker items.
        # No image decode and no global RNG consumption.
        for index, real_indices in zip(metadata['place_index'].tolist(), metadata['real_indices'].tolist()):
            selected = self.dataset.sample_paths(index, epoch=self.dataset.epoch)
            if sum(flag for _, flag in selected) != sum(i < 0 for i in real_indices):
                raise RuntimeError('Weather accounting differs from consumed synthetic slots')
            place = self.dataset.places[index]
            self.weather.update(place.condition_by_synthetic_path[path].lower() for path, synthetic in selected if synthetic)

    def summary(self):
        return {**super().summary(), 'synthetic_weather_exposures': dict(sorted(self.weather.items()))}


def train(args):
    if (args.synthetic_aux or args.teacher_checkpoint or args.reliability_ot or args.synthetic_places_only
            or args.feature_distill_weight or args.score_distill_weight or args.ot_distill_weight
            or args.real_global_distill_weight or args.real_sampling != 'coverage_pk' or args.synthetic_mode != 'mix'):
        raise ValueError('This adapter permits only full-pool coverage A/M MS training')
    original = training_data.SamplingCoverage
    training_data.SamplingCoverage = WeatherCoverage
    try:
        return train_salad.train(args)
    finally:
        training_data.SamplingCoverage = original


if __name__ == '__main__':
    train(train_salad.parse_args())
