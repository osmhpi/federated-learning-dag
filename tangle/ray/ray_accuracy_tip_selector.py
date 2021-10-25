from ..core.tip_selection import AccuracyTipSelector

class RayAccuracyTipSelector(AccuracyTipSelector):
    def __init__(self, tangle, tip_selection_settings, particle_settings):
        super().__init__(tangle, tip_selection_settings, particle_settings)
        self.precomputed_ratings = None

    def add_precomputed_ratings(self, precomputed_ratings):
        self.precomputed_ratings = precomputed_ratings

    def _compute_ratings(self, node, tx=None):
        if self.precomputed_ratings is None:
            raise RuntimeError('Variable precomputed_ratings has to be set manually by calling add_precomputed_ratings()')
        return self.precomputed_ratings
