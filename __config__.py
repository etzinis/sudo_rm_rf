import os


WSJ0_MIX_2_8K_PATH = '/mnt/data/wsj0-mix/2speakers/wav8k'
WSJ0_MIX_2_16K_PATH = '/mnt/data/wsj0-mix/2speakers/wav16k'

WSJ0_MIX_2_8K_PREPROCESSED_BASE_P = '/mnt/nvme/wsj0_mix_preprocessed'
WSJ0_MIX_2_8K_PREPROCESSED_MIN_BASE_P = os.path.join(
    WSJ0_MIX_2_8K_PREPROCESSED_BASE_P,
    'wsj0_2mix_8.0k_4.0s_min_preprocessed')
WSJ0_MIX_2_8K_PREPROCESSED_MAX_BASE_P = os.path.join(
    WSJ0_MIX_2_8K_PREPROCESSED_BASE_P,
    'wsj0_2mix_8.0k_4.0s_max_preprocessed')

WSJ_MIX_2_8K_PREPROCESSED_TRAIN_P = os.path.join(WSJ0_MIX_2_8K_PREPROCESSED_MIN_BASE_P, 'tr')
WSJ_MIX_2_8K_PREPROCESSED_EVAL_P = os.path.join(WSJ0_MIX_2_8K_PREPROCESSED_MIN_BASE_P, 'cv')
WSJ_MIX_2_8K_PREPROCESSED_TEST_P = os.path.join(WSJ0_MIX_2_8K_PREPROCESSED_MIN_BASE_P, 'tt')

TIMIT_MIX_2_8K_PREPROCESSED_TRAIN_P = \
    '/mnt/data/end2end_unsupervised_holder' \
    '/timit_10000_1000_1000_2_fm_random_taus_delays/train'
TIMIT_MIX_2_8K_PREPROCESSED_EVAL_P = \
    '/mnt/data/end2end_unsupervised_holder' \
    '/timit_10000_1000_1000_2_fm_random_taus_delays/val'
TIMIT_MIX_2_8K_PREPROCESSED_TEST_P = \
    '/mnt/data/end2end_unsupervised_holder' \
    '/timit_10000_1000_1000_2_fm_random_taus_delays/test'

WSJ_MIX_2_8K_PREPROCESSED_TRAIN_PAD_P = os.path.join(WSJ0_MIX_2_8K_PREPROCESSED_MAX_BASE_P, 'tr')
WSJ_MIX_2_8K_PREPROCESSED_EVAL_PAD_P = os.path.join(WSJ0_MIX_2_8K_PREPROCESSED_MAX_BASE_P, 'cv')
WSJ_MIX_2_8K_PREPROCESSED_TEST_PAD_P = os.path.join(WSJ0_MIX_2_8K_PREPROCESSED_MAX_BASE_P, 'tt')

WSJ_MIX_HIERARCHICAL_P = '/mnt/data/hierarchical_sound_datasets/WSJ0_mix_partitioned/'
ESC50_HIERARCHICAL_P = '/mnt/data/hierarchical_sound_datasets/ESC50_partitioned/'

AFE_WSJ_MIX_2_8K = '/home/thymios/afes/min/'
AFE_WSJ_MIX_2_8K_PAD = '/home/thymios/afes/pad/'
AFE_WSJ_MIX_2_8K_NORMPAD = '/home/thymios/afes/norm_pad/'
AFE_AUGMENTED = '/home/thymios/afes/augmented/'

TNMASK_WSJ_MIX_2_8K = '/home/thymios/tn_mask/min/'
TNMASK_WSJ_MIX_2_8K_PAD = '/home/thymios/tn_mask/pad/'
TNMASK_WSJ_MIX_2_8K_NORMPAD = '/home/thymios/tn_mask/norm_pad/'
TNMASK_AUGMENTED = '/home/thymios/tn_mask/augmented/'


API_KEY = 'your_comet_ml_key'
