import getpass
import sys

WORK_DIR = 'work'
IMPORT_PATHS = ['config', 'recipe', 'recipe/']

def TODO():
  raise Exception("Missing parameter.")

def check_engine_limits(current_rqmt, task):
    current_rqmt['time'] = min(168, current_rqmt.get('time', 2))
    return current_rqmt

def engine():
    from sisyphus.engine import EngineSelector
    from sisyphus.localengine import LocalEngine
    from sisyphus.simple_linux_utility_for_resource_management_engine import SimpleLinuxUtilityForResourceManagementEngine
    return EngineSelector(
        engines={'short': LocalEngine(cpu=4),
                 'long': SimpleLinuxUtilityForResourceManagementEngine(default_rqmt={'cpu': 1, 'mem': 4, 'time': 1})},
        default_engine='long')

# Application specific settings
PYTHON_EXE = TODO() # Add path to your python executable
IMPORT_PATHS = ['config', 'recipe/']
SIS_COMMAND = [PYTHON_EXE, sys.argv[0]]
MAX_PARALLEL = 20
CODE_ROOT = TODO()
# SGE_SSH_COMMANDS = [' source /u/standard/settings/sge_settings.sh; ']

# how many seconds should be waited before ...
WAIT_PERIOD_JOB_FS_SYNC = 30  # finishing a job
WAIT_PERIOD_BETWEEN_CHECKS = 30  # checking for finished jobs
WAIT_PERIOD_CACHE = 30  # stoping to wait for actionable jobs to appear
WAIT_PERIOD_SSH_TIMEOUT = 30  # retrying ssh connection
WAIT_PERIOD_QSTAT_PARSING = 30  # retrying to parse qstat output
WAIT_PERIOD_HTTP_RETRY_BIND = 30  # retrying to bind to the desired port
WAIT_PERIOD_JOB_CLEANUP = 30  # cleaning up a job
WAIT_PERIOD_MTIME_OF_INPUTS = 60  # wait X seconds long before starting a job to avoid file system sync problems

PRINT_ERROR_LINES = 1
SHOW_JOB_TARGETS  = False
CLEAR_ERROR = False  # set true to automatically clean jobs in error state

JOB_CLEANUP_KEEP_WORK = True
JOB_FINAL_LOG = 'finished.tar.gz'

versions = {'cuda': '10.2', 'acml': '4.4.0', 'cudnn':'7.6'}

DEFAULT_ENVIRONMENT_SET['LD_LIBRARY_PATH'] = ':'.join([
  '/usr/local/cudnn-{cuda}-v{cudnn}/lib64'.format(**versions),
  '/usr/local/cuda-{cuda}/lib64'.format(**versions),
  '/usr/local/cuda-{cuda}/extras/CUPTI/lib64'.format(**versions),
  '/usr/local/acml-{acml}/cblas_mp/lib'.format(**versions),
  '/usr/local/acml-{acml}/gfortran64/lib'.format(**versions),
  '/usr/local/acml-{acml}/gfortran64_mp/lib/'.format(**versions)
])

# DEFAULT_ENVIRONMENT_SET['PATH']            = '/usr/local/cuda-10.1/bin:' + DEFAULT_ENVIRONMENT_SET['PATH']
DEFAULT_ENVIRONMENT_SET['HDF5_USE_FILE_LOCKING']='FALSE'
CLEANUP_ENVIRONMENT = False