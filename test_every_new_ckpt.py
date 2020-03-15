from subprocess import call
from params import test_params

if  __name__ == '__main__':

    ckpt_dir = test_params.CKPT_DIR
    call(['bash', 'utils/run_every_new_ckpt.sh', ckpt_dir])
