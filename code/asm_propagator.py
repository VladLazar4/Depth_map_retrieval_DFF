import torch
import torch.nn.functional as F

class AngularSpectrumPropagator:
    """
    Initialize the ASM propagator and pre-compute the spatial frequency grid.
    """
    def __init__(self, nx: int, ny: int, pixel_pitch: float, wavelength: float, device: torch.device):
        self.nx = nx
        self.ny = ny
        self.pixel_pitch = pixel_pitch
        self.wavelength = wavelength
        self.device = device
        
        # Compute the frequences on the X and Y axis on the GPU
        fx = torch.fft.fftfreq(nx, d=pixel_pitch, device=device)
        fy = torch.fft.fftfreq(ny, d=pixel_pitch, device=device)
        
        self.FX, self.FY = torch.meshgrid(fx, fy, indexing='ij')
        
        # Squared radial frequences: fx^2 + fy^2
        self.F_squared = self.FX**2 + self.FY**2
        
        
        # Propagation mask: let only the waves that makes sense phisically (that will not cause square root of a negattive number)
        self.limit = 1.0 / (wavelength**2)
        self.prop_mask = self.F_squared <= self.limit
        
        # Z-axis frequency (f_z)
        self.FZ = torch.zeros_like(self.F_squared)
        self.FZ[self.prop_mask] = torch.sqrt(self.limit - self.F_squared[self.prop_mask])
        

    def propagate(self, H: torch.Tensor, z: float) -> torch.Tensor:
        """
        Propagate the complex hologram H to distance z
        """
        # 1. Fourier transform of the holoram
        A = torch.fft.fft2(H)
        
        # 2. Compute H transfer function (phase filter)
        phase = 2 * torch.pi * z * self.FZ
        H_transfer = torch.exp(1j * phase)
        
        H_transfer = H_transfer * self.prop_mask
        
        # 3. Apply the rotation for every wavelength (multiply the spectrum with the transfer function)
        U = A * H_transfer
        
        # 4. Inverse Fourier Transform
        H_propagated = torch.fft.ifft2(U)
        
        # The amplitude is the absolute value of H_propagated
        amplitude = torch.abs(H_propagated)

        return amplitude