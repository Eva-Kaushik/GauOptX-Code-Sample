
from .base import AcquisitionBase
from .EI import AcquisitionEI
from GauOptX.acquisitions.EI_mcmc import AcquisitionEI_MCMC
from .MPI import AcquisitionMPI
from .MPI_mcmc import AcquisitionMPI_MCMC
from .LCB import AcquisitionLCB
from .LCB_mcmc import AcquisitionLCB_MCMC
from .LP import AcquisitionLP
from .ES import AcquisitionEntropySearch

def select_acquisition(name):
    '''
    Acquisition selector for GauOptX
    '''
    if name == 'EI':
        return AcquisitionEI
    elif name == 'EI_MCMC':
        return AcquisitionEI_MCMC
    elif name == 'LCB':
        return AcquisitionLCB
    elif name == 'LCB_MCMC':
        return AcquisitionLCB_MCMC
    elif name == 'MPI':
        return AcquisitionMPI
    elif name == 'MPI_MCMC':
        return AcquisitionMPI_MCMC
    elif name == 'LP':
        return AcquisitionLP
    elif name == 'ES':
        return AcquisitionEntropySearch
    else:
        raise Exception('Invalid acquisition selected.')
