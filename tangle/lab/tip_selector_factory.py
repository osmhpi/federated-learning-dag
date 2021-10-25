from ..core.tip_selection import TipSelector, AccuracyTipSelector
from ..core.tip_selection.tip_selector import TipSelectorSettings
from ..core.tip_selection.accuracy_tip_selector import AccuracyTipSelectorSettings
from ..core.tip_selection.lazy_accuracy_tip_selector import LazyAccuracyTipSelector

class TipSelectorFactory:
    def __init__(self, config):
        self.config = config
        
        self.particle_settings = {}
        self.particle_settings[TipSelectorSettings.USE_PARTICLES] = self.config.use_particles
        self.particle_settings[TipSelectorSettings.PARTICLES_DEPTH_START] = self.config.particles_depth_start
        self.particle_settings[TipSelectorSettings.PARTICLES_DEPTH_END] = self.config.particles_depth_end
        self.particle_settings[TipSelectorSettings.NUM_PARTICLES] = self.config.particles_number

    def create(self, tangle):

        tip_selection_settings = {}
        tip_selection_settings[AccuracyTipSelectorSettings.SELECTION_STRATEGY] = self.config.acc_tip_selection_strategy
        tip_selection_settings[AccuracyTipSelectorSettings.CUMULATE_RATINGS] = self.config.acc_cumulate_ratings
        tip_selection_settings[AccuracyTipSelectorSettings.RATINGS_TO_WEIGHT] = self.config.acc_ratings_to_weights
        tip_selection_settings[AccuracyTipSelectorSettings.SELECT_FROM_WEIGHTS] = self.config.acc_select_from_weights
        tip_selection_settings[AccuracyTipSelectorSettings.ALPHA] = self.config.acc_alpha

        if self.config.tip_selector == 'default':
            return TipSelector(tangle, particle_settings=self.particle_settings)

        elif self.config.tip_selector == 'accuracy':
            return AccuracyTipSelector(tangle, tip_selection_settings, self.particle_settings)

        elif self.config.tip_selector == 'lazy_accuracy':
            return LazyAccuracyTipSelector(tangle, tip_selection_settings, self.particle_settings)
