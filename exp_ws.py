# run as follows:
# python exp_ws.py usual
# python exp_ws.py residual_usual
# etc

from wakesleep import ws
import tensorflow as tf
from utils import Struct
import sys

experiments = {}

experiments['usual'] = Struct(optimizer=tf.train.AdamOptimizer,
                              sleep_type='usual',
                              mb_size=200,
                              latent_units=[50, 20],
                              q_units=[[500] * 2, [50] * 2],
                              p_units=[[50] * 2, [500] * 2],
                              bn=True,
                              lr=0.001,
                              n_epochs=100,
                              directory='default/usual'
                              )
experiments['mix'] = experiments['usual'].updated(dict(sleep_type='mix', directory='default/mix'))

experiments['autoenc'] = experiments['usual'].updated(dict(sleep_type='autoenc', directory='default/autoenc'))

experiments['more_latents'] = experiments['usual'].updated(dict(latent_units=[100, 50], directory='more_latents/usual'))
experiments['more_latents_mix'] = experiments['more_latents'].updated(dict(sleep_type='mix', directory='more_latents/mix'))
experiments['more_latents_autoenc'] = experiments['more_latents'].updated(dict(sleep_type='autoenc', directory='more_latents/autoenc'))

experiments['three_layers'] = experiments['mix'].updated(dict(latent_units=[100, 50, 20],
                                                                q_units=[[500] * 2, [200] * 2, [50] * 2],
                                                                p_units=[[50] * 2, [200] * 2, [500] * 2],
                                                                directory='three_layers/mix'))

experiments['four_layers_mix'] = experiments['mix'].updated(dict(latent_units=[200, 100, 50, 20],
                                                                q_units=[[500] * 2, [200] * 2,  [100] * 2, [50] * 2],
                                                                p_units=[[50] * 2, [100] * 2, [200] * 2, [500] * 2],
                                                                directory='four_layers/mix'))

experiments['four_layers_usual'] = experiments['four_layers_mix'].updated(dict(sleep_type='usual',
                                                                directory='four_layers/usual'))

experiments['four_layers_autoenc'] = experiments['four_layers_mix'].updated(dict(sleep_type='autoenc',
                                                                directory='four_layers/autoenc'))

experiments['three_layers_small_mix'] = experiments['mix'].updated(dict(latent_units=[100, 50, 10],
                                                                q_units=[[500] * 2, [200] * 2, [50] * 2],
                                                                p_units=[[50] * 2, [200] * 2, [500] * 2],
                                                                directory='three_layers_small/mix'))
experiments['three_layers_small_usual'] = experiments['three_layers_small_mix'].updated(dict(sleep_type='usual',
                                                                directory='three_layers_small/usual'))
experiments['three_layers_small_autoenc'] = experiments['three_layers_small_mix'].updated(dict(sleep_type='autoenc',
                                                                directory='three_layers_small/autoenc'))


experiments['three_layers_usual'] = experiments['three_layers'].updated(dict(sleep_type='usual',
                                                                directory='three_layers/usual'))
experiments['three_layers_autoenc'] = experiments['three_layers'].updated(dict(sleep_type='autoenc',
                                                                directory='three_layers/autoenc'))
experiments['three_layers_mixy_mix'] = experiments['three_layers'].updated(dict(sleep_type='mixy_mix',
                                                                directory='three_layers/_mixy_mix'))
experiments['three_layers_top_layer_mix'] = experiments['three_layers'].updated(dict(sleep_type='top_layer_mix',
                                                                directory='three_layers/top_layer_mix'))
experiments['three_layers_top_down_mix1'] = experiments['three_layers'].updated(dict(sleep_type='top_down_mix1',
                                                                directory='three_layers/top_down_mix1'))
experiments['three_layers_top_down_mix2'] = experiments['three_layers'].updated(dict(sleep_type='top_down_mix2',
                                                                directory='three_layers/top_down_mix2'))


experiments['three_layers_small_p'] = experiments['mix'].updated(dict(latent_units=[200, 200, 200],
                                                                q_units=[[500] * 2, [200] * 2, [200] * 2],
                                                                p_units=[[], [], []],
                                                                directory='three_layers_small_p/mix'))


experiments['very_deep'] = experiments['mix'].updated(dict(latent_units=[200]*9,
                                                                q_units=[[200] * 2]*9,
                                                                p_units=[[]]*9,
                                                                directory='very_deep/mix'))
experiments['very_deep_usual'] = experiments['very_deep'].updated(dict(sleep_type='usual', directory='very_deep/usual'))
experiments['very_deep_autoenc'] = experiments['very_deep'].updated(dict(sleep_type='autoenc', directory='very_deep/autoenc'))


experiments['very_deep_small_q'] = experiments['mix'].updated(dict(latent_units=[200]*9,
                                                                q_units=[[]]*9,
                                                                p_units=[[]]*9,
                                                                directory='very_deep_small_q/mix'))
experiments['very_deep_small_q_usual'] = experiments['very_deep_small_q'].updated(dict(sleep_type='usual', directory='very_deep_small_q/usual'))

experiments['small_q'] = experiments['mix'].updated(dict(latent_units=[200]*3,
                                                                q_units=[[]]*3,
                                                                p_units=[[200] * 2]*3,
                                                                directory='small_q/mix'))

experiments['200_200_mix'] = experiments['mix'].updated(dict(latent_units=[200]*2,
                                                                q_units=[[]]*2,
                                                                p_units=[[]]*2,
                                                                directory='200_200/mix'))
experiments['200_200_usual'] = experiments['200_200_mix'].updated(dict(sleep_type='usual', directory='200_200/usual'))

experiments['stochastic5'] = experiments['usual'].updated(dict(latent_units=[400, 300, 200, 100, 10],
                                                                q_units=[[]]*5,
                                                                p_units=[[]]*5,
                                                                directory='stochastic5/usual'))
experiments['stochastic5_mix'] = experiments['stochastic5'].updated(dict(sleep_type='mix', directory='stochastic5/mix'))
experiments['stochastic5_autoenc'] = experiments['stochastic5'].updated(dict(sleep_type='autoenc', directory='stochastic5/autoenc'))
experiments['stochastic5_top_down_mix1'] = experiments['stochastic5'].updated(dict(sleep_type='top_down_mix1', directory='stochastic5/top_down_mix1'))

experiments['small'] = experiments['usual'].updated(dict(latent_units=[200]*3,
                                                                q_units=[[]]*3,
                                                                p_units=[[]]*3,
                                                                directory='small/usual'))
experiments['small_mix'] = experiments['small'].updated(dict(sleep_type='mix', directory='small/mix'))
experiments['small_autoenc'] = experiments['small'].updated(dict(sleep_type='autoenc', directory='small/autoenc'))
experiments['small_top_down_mix1'] = experiments['small'].updated(dict(sleep_type='top_down_mix1', directory='small/top_down_mix1'))




if __name__ == '__main__':
    exp = sys.argv[1]
    exp = experiments[exp]
    if len(sys.argv) > 2 and sys.argv[2] in ['-r', 'restore']:
        exp = exp.updated(dict(restore=True))
    ws.main(exp)
