import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any, Union
import torch
import torch.nn as nn

# =============================================================================
# PYTORCH-BASED SPECTRUM FITTING
# =============================================================================

class LorentzianFitter:
    """
    PyTorch-based multi-peak Lorentzian fitting with batch processing
    
    This class implements flexible N-peak Lorentzian fitting using:
    - PyTorch autograd for gradient-based optimization
    - RAdam optimizer for stable convergence
    - Batch processing for multiple spectra simultaneously
    - Soft constraints via regularization
    
    Physical Model:
    ---------------
    Lorentzian: I(λ) = Σ (2*a_i/π) * (c_i / (4*(λ-b_i)² + c_i²))
    
    where for each peak i:
    - a_i: amplitude (height)
    - b_i: position (wavelength in nm)
    - c_i: FWHM (full width at half maximum in nm)
    """
    
    def __init__(self, args: Dict[str, Any], use_gpu: bool = False):
        """
        Initialize Lorentzian fitter
        
        Parameters:
        -----------
        args : Dict[str, Any]
            Configuration dictionary
        use_gpu : bool
            Whether to use GPU acceleration
        """
        self.args = args
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.num_peaks = args['NUM_PEAKS']
        
    def lorentzian_function(self, 
                           wavelengths: torch.Tensor,
                           amplitudes: torch.Tensor,
                           positions: torch.Tensor,
                           widths: torch.Tensor) -> torch.Tensor:
        """
        Evaluate N-peak Lorentzian function
        
        Parameters:
        -----------
        wavelengths : torch.Tensor, shape (N_batch, N_wl)
        amplitudes : torch.Tensor, shape (N_batch, N_peaks)
        positions : torch.Tensor, shape (N_batch, N_peaks)
        widths : torch.Tensor, shape (N_batch, N_peaks)
        
        Returns:
        --------
        torch.Tensor, shape (N_batch, N_wl)
            Sum of N Lorentzian peaks
        """
        N_batch, N_wl = wavelengths.shape
        N_peaks = amplitudes.shape[1]
        
        # Expand dimensions for broadcasting
        # wavelengths: (N_batch, N_wl) -> (N_batch, N_peaks, N_wl)
        wl = wavelengths.unsqueeze(1).expand(N_batch, N_peaks, N_wl)
        
        # positions, widths, amplitudes: (N_batch, N_peaks) -> (N_batch, N_peaks, 1)
        pos = positions.unsqueeze(2)
        w = widths.unsqueeze(2)
        a = amplitudes.unsqueeze(2)
        
        # Lorentzian formula
        lorentz = (2 * a / np.pi) * (w / (4 * (wl - pos).pow(2) + w.pow(2)))
        
        # Sum over peaks
        result = lorentz.sum(dim=1)  # (N_batch, N_wl)
        
        return result
    
    def initialize_parameters(self, 
                             spectra: np.ndarray,
                             wavelengths: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize fitting parameters from user-provided initial guesses
        
        Parameters:
        -----------
        spectra : np.ndarray, shape (N_batch, N_wl)
        wavelengths : np.ndarray, shape (N_wl,)
        
        Returns:
        --------
        Tuple of (positions, widths, amplitudes) as torch tensors
        """
        N_batch = spectra.shape[0]
        
        # Get user-provided initial guesses
        pos_init = self.args['PEAK_POSITION_INITIAL_GUESS']
        width_init = self.args['PEAK_WIDTH_INITIAL_GUESS']
        height_init = self.args['PEAK_HEIGHT_INITIAL_GUESS']
        
        # Validate lengths
        if len(pos_init) != self.num_peaks:
            raise ValueError(f"PEAK_POSITION_INITIAL_GUESS length ({len(pos_init)}) must match NUM_PEAKS ({self.num_peaks})")
        if len(width_init) != self.num_peaks:
            raise ValueError(f"PEAK_WIDTH_INITIAL_GUESS length ({len(width_init)}) must match NUM_PEAKS ({self.num_peaks})")
        if len(height_init) != self.num_peaks:
            raise ValueError(f"PEAK_HEIGHT_INITIAL_GUESS length ({len(height_init)}) must match NUM_PEAKS ({self.num_peaks})")
        
        # Convert to tensors and repeat for each spectrum in batch
        positions = torch.tensor(pos_init, dtype=torch.float32, device=self.device)
        widths = torch.tensor(width_init, dtype=torch.float32, device=self.device)
        heights = torch.tensor(height_init, dtype=torch.float32, device=self.device)
        
        # Expand to batch: (N_peaks,) -> (N_batch, N_peaks)
        positions = positions.unsqueeze(0).expand(N_batch, self.num_peaks).clone()
        widths = widths.unsqueeze(0).expand(N_batch, self.num_peaks).clone()
        heights = heights.unsqueeze(0).expand(N_batch, self.num_peaks).clone()
        
        return positions, widths, heights
    
    def compute_regularization(self,
                              positions: torch.Tensor,
                              widths: torch.Tensor,
                              heights: torch.Tensor,
                              pos_init: torch.Tensor) -> torch.Tensor:
        """
        Compute regularization losses for soft constraints
        
        Parameters:
        -----------
        positions : torch.Tensor, shape (N_batch, N_peaks)
        widths : torch.Tensor, shape (N_batch, N_peaks)
        heights : torch.Tensor, shape (N_batch, N_peaks)
        pos_init : torch.Tensor, shape (N_batch, N_peaks)
        
        Returns:
        --------
        torch.Tensor, scalar
            Total regularization loss
        """
        reg_loss = torch.tensor(0.0, device=self.device)
        
        # 1. Negative height penalty
        reg_height = self.args.get('REG_NEGATIVE_HEIGHT', 1.0)
        if reg_height > 0:
            negative_heights = torch.relu(-heights)  # Only penalize if < 0
            reg_loss = reg_loss + reg_height * negative_heights.pow(2).sum()
        
        # 2. Width maximum constraint (soft)
        width_max = self.args.get('PEAK_WIDTH_MAX', None)
        reg_width_max = self.args.get('REG_WIDTH_MAX', 0.1)
        if width_max is not None and reg_width_max > 0:
            if isinstance(width_max, (list, tuple)):
                width_max_tensor = torch.tensor(width_max, dtype=torch.float32, device=self.device)
                width_max_tensor = width_max_tensor.unsqueeze(0).expand_as(widths)
            else:
                width_max_tensor = torch.tensor(width_max, dtype=torch.float32, device=self.device)
            
            excess_width = torch.relu(widths - width_max_tensor)
            reg_loss = reg_loss + reg_width_max * excess_width.pow(2).sum()
        
        # 3. Position constraint (stay within tolerance of initial guess)
        pos_tol = self.args.get('PEAK_POSITION_TOLERANCE', None)
        reg_pos = self.args.get('REG_POSITION_CONSTRAINT', 1.0)
        if pos_tol is not None and reg_pos > 0:
            if isinstance(pos_tol, (list, tuple)):
                pos_tol_tensor = torch.tensor(pos_tol, dtype=torch.float32, device=self.device)
                pos_tol_tensor = pos_tol_tensor.unsqueeze(0).expand_as(positions)
            else:
                pos_tol_tensor = torch.tensor(pos_tol, dtype=torch.float32, device=self.device)
            
            pos_deviation = torch.abs(positions - pos_init)
            excess_deviation = torch.relu(pos_deviation - pos_tol_tensor)
            reg_loss = reg_loss + reg_pos * excess_deviation.pow(2).sum()
        
        # 4. Multi-peak: minimum distance between peaks
        if self.num_peaks >= 2:
            min_distance = self.args.get('PEAK_MIN_DISTANCE', None)
            reg_distance = self.args.get('REG_PEAK_DISTANCE', 0.01)
            if min_distance is not None and reg_distance > 0:
                # For each pair of peaks, check distance
                for i in range(self.num_peaks):
                    for j in range(i + 1, self.num_peaks):
                        distance = torch.abs(positions[:, i] - positions[:, j])
                        violation = torch.relu(min_distance - distance)
                        reg_loss = reg_loss + reg_distance * violation.pow(2).sum()
        
        return reg_loss
    
    def fit(self, 
            spectra: np.ndarray,
            wavelengths: np.ndarray) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        """
        Fit multiple spectra simultaneously using batch processing
        
        Parameters:
        -----------
        spectra : np.ndarray, shape (N_batch, N_wl)
            Batch of spectra to fit
        wavelengths : np.ndarray, shape (N_wl,)
            Wavelength array
        
        Returns:
        --------
        Tuple of:
            - fitted_spectra: np.ndarray, shape (N_batch, N_wl)
            - params_list: List of N_batch parameter dictionaries
            - r2_scores: np.ndarray, shape (N_batch,)
        """
        N_batch = spectra.shape[0]
        
        # Get fitting range
        fit_min, fit_max = self.args['FIT_RANGE_NM']
        mask = (wavelengths >= fit_min) & (wavelengths <= fit_max)
        wl_fit = wavelengths[mask]
        spectra_fit = spectra[:, mask]
        
        # Convert to tensors
        wl_tensor = torch.tensor(wl_fit, dtype=torch.float32, device=self.device)
        wl_tensor = wl_tensor.unsqueeze(0).expand(N_batch, -1)  # (N_batch, N_wl)
        
        spectra_tensor = torch.tensor(spectra_fit, dtype=torch.float32, device=self.device)
        
        # Initialize parameters
        positions, widths, heights = self.initialize_parameters(spectra, wavelengths)
        pos_init = positions.clone()  # Store for regularization
        
        # Make parameters learnable
        positions = nn.Parameter(positions)
        widths = nn.Parameter(widths)
        heights = nn.Parameter(heights)
        
        # Setup optimizer
        optimizer_name = self.args.get('OPTIMIZER', 'RAdam')
        lr = self.args.get('LEARNING_RATE', 0.01)
        num_iter = self.args.get('NUM_ITERATIONS', 1000)
        
        if optimizer_name == 'RAdam':
            optimizer = torch.optim.RAdam([positions, widths, heights], lr=lr)
        elif optimizer_name == 'Adam':
            optimizer = torch.optim.Adam([positions, widths, heights], lr=lr)
        elif optimizer_name == 'NAdam':
            optimizer = torch.optim.NAdam([positions, widths, heights], lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Optional: Learning rate scheduler
        use_scheduler = self.args.get('USE_LR_SCHEDULER', False)
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=100, verbose=False
            )
        
        # Optimization loop
        print_every = self.args.get('PRINT_EVERY', 100)
        
        for iteration in range(num_iter):
            optimizer.zero_grad()
            
            # Forward pass
            fitted = self.lorentzian_function(wl_tensor, heights, positions, widths)
            
            # MSE loss
            mse_loss = (fitted - spectra_tensor).pow(2).mean()
            
            # Regularization
            reg_loss = self.compute_regularization(positions, widths, heights, pos_init)
            
            # Total loss
            total_loss = mse_loss + reg_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Update scheduler
            if use_scheduler:
                scheduler.step(total_loss)
            
            # Print progress
            if (iteration + 1) % print_every == 0 or iteration == 0:
                print(f"[info] Iteration {iteration + 1}/{num_iter}: "
                      f"MSE={mse_loss.item():.6e}, Reg={reg_loss.item():.6e}, "
                      f"Total={total_loss.item():.6e}")
        
        # Generate full fitted curves (over entire wavelength range)
        with torch.no_grad():
            wl_full = torch.tensor(wavelengths, dtype=torch.float32, device=self.device)
            wl_full = wl_full.unsqueeze(0).expand(N_batch, -1)
            
            fitted_full = self.lorentzian_function(wl_full, heights, positions, widths)
            
            # Convert back to numpy
            fitted_full_np = fitted_full.cpu().numpy()
            positions_np = positions.detach().cpu().numpy()
            widths_np = widths.detach().cpu().numpy()
            heights_np = heights.detach().cpu().numpy()
        
        # Compute R² for each spectrum
        spectra_full_tensor = torch.tensor(spectra, dtype=torch.float32, device=self.device)
        r2_scores = np.zeros(N_batch)
        
        with torch.no_grad():
            for i in range(N_batch):
                ss_res = ((spectra_full_tensor[i] - fitted_full[i]).pow(2)).sum().item()
                ss_tot = ((spectra_full_tensor[i] - spectra_full_tensor[i].mean()).pow(2)).sum().item()
                r2_scores[i] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Build parameter dictionaries
        params_list = []
        for i in range(N_batch):
            params = {}
            for peak_idx in range(self.num_peaks):
                peak_num = peak_idx + 1
                params[f'a{peak_num}'] = heights_np[i, peak_idx]
                params[f'b{peak_num}'] = positions_np[i, peak_idx]
                params[f'c{peak_num}'] = widths_np[i, peak_idx]
            
            # MATLAB compatibility for single peak
            if self.num_peaks == 1:
                params['a'] = heights_np[i, 0]
                params['b1'] = positions_np[i, 0]
                params['c1'] = widths_np[i, 0]
            
            params_list.append(params)
        
        return fitted_full_np, params_list, r2_scores


class FanoFitter:
    """
    PyTorch-based Fano resonance fitting with batch processing
    
    Physical Model:
    ---------------
    Bright mode: A_bright^(i) = c_i × (γ_i/2) / (λ - λ_i + i×γ_i/2)  [phase = 0]
    Dark mode:   A_dark^(j) = d_j × exp(i×θ_j) × (Γ_j/2) / (λ - λ_j + i×Γ_j/2)
    Total:       I(λ) = |Σ A_bright^(i) + Σ A_dark^(j)|²
    """
    
    def __init__(self, args: Dict[str, Any], use_gpu: bool = False):
        """Initialize Fano fitter"""
        self.args = args
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.num_bright = args.get('NUM_BRIGHT_MODES', 0)
        self.num_dark = args.get('NUM_DARK_MODES', 0)
        
        if self.num_bright == 0 and self.num_dark == 0:
            raise ValueError("At least one bright or dark mode must be specified")
    
    def fano_function(self,
                     wavelengths: torch.Tensor,
                     bright_c: torch.Tensor,
                     bright_pos: torch.Tensor,
                     bright_gamma: torch.Tensor,
                     dark_d: torch.Tensor,
                     dark_pos: torch.Tensor,
                     dark_Gamma: torch.Tensor,
                     dark_theta: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Fano resonance model
        
        Parameters:
        -----------
        wavelengths : torch.Tensor, shape (N_batch, N_wl)
        bright_c : torch.Tensor, shape (N_batch, N_bright)
        bright_pos : torch.Tensor, shape (N_batch, N_bright)
        bright_gamma : torch.Tensor, shape (N_batch, N_bright)
        dark_d : torch.Tensor, shape (N_batch, N_dark)
        dark_pos : torch.Tensor, shape (N_batch, N_dark)
        dark_Gamma : torch.Tensor, shape (N_batch, N_dark)
        dark_theta : torch.Tensor, shape (N_batch, N_dark)
        
        Returns:
        --------
        torch.Tensor, shape (N_batch, N_wl)
            Fano resonance intensity
        """
        N_batch, N_wl = wavelengths.shape
        
        # Initialize complex amplitude
        A_total = torch.zeros(N_batch, N_wl, dtype=torch.complex64, device=self.device)
        
        # Add bright modes (phase = 0)
        if self.num_bright > 0:
            for i in range(self.num_bright):
                c = bright_c[:, i].unsqueeze(1)  # (N_batch, 1)
                lam = bright_pos[:, i].unsqueeze(1)
                gamma = bright_gamma[:, i].unsqueeze(1)
                
                denominator = wavelengths - lam + 1j * gamma / 2
                A_bright = c * (gamma / 2) / denominator
                A_total = A_total + A_bright
        
        # Add dark modes (with phase)
        if self.num_dark > 0:
            for j in range(self.num_dark):
                d = dark_d[:, j].unsqueeze(1)
                lam = dark_pos[:, j].unsqueeze(1)
                Gamma = dark_Gamma[:, j].unsqueeze(1)
                theta = dark_theta[:, j].unsqueeze(1)
                
                phase_factor = torch.exp(1j * theta)
                denominator = wavelengths - lam + 1j * Gamma / 2
                A_dark = d * phase_factor * (Gamma / 2) / denominator
                A_total = A_total + A_dark
        
        # Intensity
        intensity = torch.abs(A_total) ** 2
        
        return intensity.real
    
    def initialize_parameters(self,
                             spectra: np.ndarray,
                             wavelengths: np.ndarray) -> Tuple:
        """Initialize Fano parameters from user-provided guesses"""
        N_batch = spectra.shape[0]
        
        # Bright modes
        if self.num_bright > 0:
            bright_pos_init = self.args['BRIGHT_POSITION_INITIAL_GUESS']
            bright_gamma_init = self.args['BRIGHT_WIDTH_INITIAL_GUESS']
            bright_c_init = self.args['BRIGHT_HEIGHT_INITIAL_GUESS']
            
            if len(bright_pos_init) != self.num_bright:
                raise ValueError(f"BRIGHT_POSITION_INITIAL_GUESS length mismatch")
            if len(bright_gamma_init) != self.num_bright:
                raise ValueError(f"BRIGHT_WIDTH_INITIAL_GUESS length mismatch")
            if len(bright_c_init) != self.num_bright:
                raise ValueError(f"BRIGHT_HEIGHT_INITIAL_GUESS length mismatch")
            
            bright_pos = torch.tensor(bright_pos_init, dtype=torch.float32, device=self.device)
            bright_gamma = torch.tensor(bright_gamma_init, dtype=torch.float32, device=self.device)
            bright_c = torch.tensor(bright_c_init, dtype=torch.float32, device=self.device)
            
            bright_pos = bright_pos.unsqueeze(0).expand(N_batch, -1).clone()
            bright_gamma = bright_gamma.unsqueeze(0).expand(N_batch, -1).clone()
            bright_c = bright_c.unsqueeze(0).expand(N_batch, -1).clone()
        else:
            bright_pos = torch.zeros(N_batch, 1, device=self.device)
            bright_gamma = torch.zeros(N_batch, 1, device=self.device)
            bright_c = torch.zeros(N_batch, 1, device=self.device)
        
        # Dark modes
        if self.num_dark > 0:
            dark_pos_init = self.args['DARK_POSITION_INITIAL_GUESS']
            dark_Gamma_init = self.args['DARK_WIDTH_INITIAL_GUESS']
            dark_d_init = self.args['DARK_HEIGHT_INITIAL_GUESS']
            dark_theta_init = self.args.get('DARK_PHASE_INITIAL_GUESS', [0.0] * self.num_dark)
            
            if len(dark_pos_init) != self.num_dark:
                raise ValueError(f"DARK_POSITION_INITIAL_GUESS length mismatch")
            if len(dark_Gamma_init) != self.num_dark:
                raise ValueError(f"DARK_WIDTH_INITIAL_GUESS length mismatch")
            if len(dark_d_init) != self.num_dark:
                raise ValueError(f"DARK_HEIGHT_INITIAL_GUESS length mismatch")
            if len(dark_theta_init) != self.num_dark:
                raise ValueError(f"DARK_PHASE_INITIAL_GUESS length mismatch")
            
            dark_pos = torch.tensor(dark_pos_init, dtype=torch.float32, device=self.device)
            dark_Gamma = torch.tensor(dark_Gamma_init, dtype=torch.float32, device=self.device)
            dark_d = torch.tensor(dark_d_init, dtype=torch.float32, device=self.device)
            dark_theta = torch.tensor(dark_theta_init, dtype=torch.float32, device=self.device)
            
            dark_pos = dark_pos.unsqueeze(0).expand(N_batch, -1).clone()
            dark_Gamma = dark_Gamma.unsqueeze(0).expand(N_batch, -1).clone()
            dark_d = dark_d.unsqueeze(0).expand(N_batch, -1).clone()
            dark_theta = dark_theta.unsqueeze(0).expand(N_batch, -1).clone()
        else:
            dark_pos = torch.zeros(N_batch, 1, device=self.device)
            dark_Gamma = torch.zeros(N_batch, 1, device=self.device)
            dark_d = torch.zeros(N_batch, 1, device=self.device)
            dark_theta = torch.zeros(N_batch, 1, device=self.device)
        
        return bright_c, bright_pos, bright_gamma, dark_d, dark_pos, dark_Gamma, dark_theta
    
    def compute_regularization(self, bright_c, bright_pos, bright_gamma, bright_pos_init,
                              dark_d, dark_pos, dark_Gamma, dark_pos_init) -> torch.Tensor:
        """Compute regularization for Fano fitting"""
        reg_loss = torch.tensor(0.0, device=self.device)
        
        # 1. Negative height penalty (bright and dark)
        reg_height = self.args.get('REG_NEGATIVE_HEIGHT', 1.0)
        if reg_height > 0:
            if self.num_bright > 0:
                reg_loss = reg_loss + reg_height * torch.relu(-bright_c).pow(2).sum()
            if self.num_dark > 0:
                reg_loss = reg_loss + reg_height * torch.relu(-dark_d).pow(2).sum()
        
        # 2. Width constraints
        reg_width_max = self.args.get('REG_WIDTH_MAX', 0.1)
        
        # Bright width max
        if self.num_bright > 0:
            bright_width_max = self.args.get('BRIGHT_WIDTH_MAX', None)
            if bright_width_max is not None and reg_width_max > 0:
                if isinstance(bright_width_max, (list, tuple)):
                    bright_max_tensor = torch.tensor(bright_width_max, dtype=torch.float32, device=self.device)
                    bright_max_tensor = bright_max_tensor.unsqueeze(0).expand_as(bright_gamma)
                else:
                    bright_max_tensor = torch.tensor(bright_width_max, dtype=torch.float32, device=self.device)
                
                excess = torch.relu(bright_gamma - bright_max_tensor)
                reg_loss = reg_loss + reg_width_max * excess.pow(2).sum()
        
        # Dark width max
        if self.num_dark > 0:
            dark_width_max = self.args.get('DARK_WIDTH_MAX', None)
            if dark_width_max is not None and reg_width_max > 0:
                if isinstance(dark_width_max, (list, tuple)):
                    dark_max_tensor = torch.tensor(dark_width_max, dtype=torch.float32, device=self.device)
                    dark_max_tensor = dark_max_tensor.unsqueeze(0).expand_as(dark_Gamma)
                else:
                    dark_max_tensor = torch.tensor(dark_width_max, dtype=torch.float32, device=self.device)
                
                excess = torch.relu(dark_Gamma - dark_max_tensor)
                reg_loss = reg_loss + reg_width_max * excess.pow(2).sum()
        
        # 3. Position constraints
        reg_pos = self.args.get('REG_POSITION_CONSTRAINT', 1.0)
        
        # Bright position tolerance
        if self.num_bright > 0:
            bright_tol = self.args.get('BRIGHT_POSITION_TOLERANCE', None)
            if bright_tol is not None and reg_pos > 0:
                if isinstance(bright_tol, (list, tuple)):
                    tol_tensor = torch.tensor(bright_tol, dtype=torch.float32, device=self.device)
                    tol_tensor = tol_tensor.unsqueeze(0).expand_as(bright_pos)
                else:
                    tol_tensor = torch.tensor(bright_tol, dtype=torch.float32, device=self.device)
                
                deviation = torch.abs(bright_pos - bright_pos_init)
                excess = torch.relu(deviation - tol_tensor)
                reg_loss = reg_loss + reg_pos * excess.pow(2).sum()
        
        # Dark position tolerance
        if self.num_dark > 0:
            dark_tol = self.args.get('DARK_POSITION_TOLERANCE', None)
            if dark_tol is not None and reg_pos > 0:
                if isinstance(dark_tol, (list, tuple)):
                    tol_tensor = torch.tensor(dark_tol, dtype=torch.float32, device=self.device)
                    tol_tensor = tol_tensor.unsqueeze(0).expand_as(dark_pos)
                else:
                    tol_tensor = torch.tensor(dark_tol, dtype=torch.float32, device=self.device)
                
                deviation = torch.abs(dark_pos - dark_pos_init)
                excess = torch.relu(deviation - tol_tensor)
                reg_loss = reg_loss + reg_pos * excess.pow(2).sum()
        
        return reg_loss
    
    def fit(self,
            spectra: np.ndarray,
            wavelengths: np.ndarray) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
        """
        Fit Fano model to multiple spectra with two-stage optimization
        
        Stage 1: Fit bright modes only
        Stage 2: Add dark modes (if any)
        """
        N_batch = spectra.shape[0]
        
        # Get fitting range
        fit_min, fit_max = self.args['FIT_RANGE_NM']
        mask = (wavelengths >= fit_min) & (wavelengths <= fit_max)
        wl_fit = wavelengths[mask]
        spectra_fit = spectra[:, mask]
        
        # Convert to tensors
        wl_tensor = torch.tensor(wl_fit, dtype=torch.float32, device=self.device)
        wl_tensor = wl_tensor.unsqueeze(0).expand(N_batch, -1)
        spectra_tensor = torch.tensor(spectra_fit, dtype=torch.float32, device=self.device)
        
        # Initialize parameters
        bright_c, bright_pos, bright_gamma, dark_d, dark_pos, dark_Gamma, dark_theta = \
            self.initialize_parameters(spectra, wavelengths)
        
        bright_pos_init = bright_pos.clone()
        dark_pos_init = dark_pos.clone()
        
        # ====================
        # STAGE 1: Bright modes only
        # ====================
        print("[info] Stage 1: Fitting bright modes...")
        
        bright_c_param = nn.Parameter(bright_c)
        bright_pos_param = nn.Parameter(bright_pos)
        bright_gamma_param = nn.Parameter(bright_gamma)
        
        optimizer_name = self.args.get('OPTIMIZER', 'RAdam')
        lr_bright = self.args.get('LEARNING_RATE_BRIGHT', 0.01)
        num_iter_bright = self.args.get('NUM_ITERATIONS_BRIGHT', 1000)
        
        if optimizer_name == 'RAdam':
            optimizer = torch.optim.RAdam([bright_c_param, bright_pos_param, bright_gamma_param], lr=lr_bright)
        elif optimizer_name == 'Adam':
            optimizer = torch.optim.Adam([bright_c_param, bright_pos_param, bright_gamma_param], lr=lr_bright)
        elif optimizer_name == 'NAdam':
            optimizer = torch.optim.NAdam([bright_c_param, bright_pos_param, bright_gamma_param], lr=lr_bright)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        print_every = self.args.get('PRINT_EVERY', 100)
        
        for iteration in range(num_iter_bright):
            optimizer.zero_grad()
            
            # Forward (bright only)
            fitted = self.fano_function(
                wl_tensor, 
                bright_c_param, bright_pos_param, bright_gamma_param,
                torch.zeros_like(dark_d), dark_pos, dark_Gamma, dark_theta
            )
            
            mse_loss = (fitted - spectra_tensor).pow(2).mean()
            reg_loss = self.compute_regularization(
                bright_c_param, bright_pos_param, bright_gamma_param, bright_pos_init,
                torch.zeros_like(dark_d), dark_pos, dark_Gamma, dark_pos_init
            )
            
            total_loss = mse_loss + reg_loss
            total_loss.backward()
            optimizer.step()
            
            if (iteration + 1) % print_every == 0 or iteration == 0:
                print(f"[info] Bright Stage {iteration + 1}/{num_iter_bright}: "
                      f"MSE={mse_loss.item():.6e}, Reg={reg_loss.item():.6e}")
        
        # ====================
        # STAGE 2: Add dark modes
        # ====================
        if self.num_dark > 0:
            print("[info] Stage 2: Adding dark modes...")
            
            # Fix bright modes, optimize dark modes
            bright_c_fixed = bright_c_param.detach().clone()
            bright_pos_fixed = bright_pos_param.detach().clone()
            bright_gamma_fixed = bright_gamma_param.detach().clone()
            
            dark_d_param = nn.Parameter(dark_d)
            dark_pos_param = nn.Parameter(dark_pos)
            dark_Gamma_param = nn.Parameter(dark_Gamma)
            dark_theta_param = nn.Parameter(dark_theta)
            
            lr_dark = self.args.get('LEARNING_RATE_DARK', 0.001)
            num_iter_dark = self.args.get('NUM_ITERATIONS_DARK', 1000)
            
            if optimizer_name == 'RAdam':
                optimizer = torch.optim.RAdam([dark_d_param, dark_pos_param, dark_Gamma_param, dark_theta_param], lr=lr_dark)
            elif optimizer_name == 'Adam':
                optimizer = torch.optim.Adam([dark_d_param, dark_pos_param, dark_Gamma_param, dark_theta_param], lr=lr_dark)
            elif optimizer_name == 'NAdam':
                optimizer = torch.optim.NAdam([dark_d_param, dark_pos_param, dark_Gamma_param, dark_theta_param], lr=lr_dark)
            
            for iteration in range(num_iter_dark):
                optimizer.zero_grad()
                
                fitted = self.fano_function(
                    wl_tensor,
                    bright_c_fixed, bright_pos_fixed, bright_gamma_fixed,
                    dark_d_param, dark_pos_param, dark_Gamma_param, dark_theta_param
                )
                
                mse_loss = (fitted - spectra_tensor).pow(2).mean()
                reg_loss = self.compute_regularization(
                    bright_c_fixed, bright_pos_fixed, bright_gamma_fixed, bright_pos_init,
                    dark_d_param, dark_pos_param, dark_Gamma_param, dark_pos_init
                )
                
                total_loss = mse_loss + reg_loss
                total_loss.backward()
                optimizer.step()
                
                if (iteration + 1) % print_every == 0 or iteration == 0:
                    print(f"[info] Dark Stage {iteration + 1}/{num_iter_dark}: "
                          f"MSE={mse_loss.item():.6e}, Reg={reg_loss.item():.6e}")
            
            # Use final parameters
            bright_c_final = bright_c_fixed
            bright_pos_final = bright_pos_fixed
            bright_gamma_final = bright_gamma_fixed
            dark_d_final = dark_d_param
            dark_pos_final = dark_pos_param
            dark_Gamma_final = dark_Gamma_param
            dark_theta_final = dark_theta_param
        else:
            # Only bright modes
            bright_c_final = bright_c_param
            bright_pos_final = bright_pos_param
            bright_gamma_final = bright_gamma_param
            dark_d_final = dark_d
            dark_pos_final = dark_pos
            dark_Gamma_final = dark_Gamma
            dark_theta_final = dark_theta
        
        # Generate full fitted curves
        with torch.no_grad():
            wl_full = torch.tensor(wavelengths, dtype=torch.float32, device=self.device)
            wl_full = wl_full.unsqueeze(0).expand(N_batch, -1)
            
            fitted_full = self.fano_function(
                wl_full,
                bright_c_final, bright_pos_final, bright_gamma_final,
                dark_d_final, dark_pos_final, dark_Gamma_final, dark_theta_final
            )
            
            fitted_full_np = fitted_full.cpu().numpy()
        
        # Compute R²
        spectra_full_tensor = torch.tensor(spectra, dtype=torch.float32, device=self.device)
        r2_scores = np.zeros(N_batch)
        
        with torch.no_grad():
            for i in range(N_batch):
                ss_res = ((spectra_full_tensor[i] - fitted_full[i]).pow(2)).sum().item()
                ss_tot = ((spectra_full_tensor[i] - spectra_full_tensor[i].mean()).pow(2)).sum().item()
                r2_scores[i] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Build parameter dictionaries
        bright_c_np = bright_c_final.detach().cpu().numpy()
        bright_pos_np = bright_pos_final.detach().cpu().numpy()
        bright_gamma_np = bright_gamma_final.detach().cpu().numpy()
        dark_d_np = dark_d_final.detach().cpu().numpy()
        dark_pos_np = dark_pos_final.detach().cpu().numpy()
        dark_Gamma_np = dark_Gamma_final.detach().cpu().numpy()
        dark_theta_np = dark_theta_final.detach().cpu().numpy()
        
        params_list = []
        for i in range(N_batch):
            params = {}
            
            # Bright modes
            for j in range(self.num_bright):
                params[f'bright{j+1}_c'] = bright_c_np[i, j]
                params[f'bright{j+1}_lambda'] = bright_pos_np[i, j]
                params[f'bright{j+1}_gamma'] = bright_gamma_np[i, j]
            
            # Dark modes
            for j in range(self.num_dark):
                params[f'dark{j+1}_d'] = dark_d_np[i, j]
                params[f'dark{j+1}_lambda'] = dark_pos_np[i, j]
                params[f'dark{j+1}_Gamma'] = dark_Gamma_np[i, j]
                params[f'dark{j+1}_theta'] = dark_theta_np[i, j]
            
            params_list.append(params)
        
        return fitted_full_np, params_list, r2_scores


# =============================================================================
# PUBLIC API FUNCTIONS
# =============================================================================

def fit_lorentz_batch(args: Dict[str, Any],
                      spectra: np.ndarray,
                      wavelengths: np.ndarray,
                      use_gpu: bool = False) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
    """
    Fit Lorentzian model to multiple spectra using PyTorch batch processing
    
    Parameters:
    -----------
    args : Dict[str, Any]
        Configuration dictionary containing:
        - NUM_PEAKS: Number of peaks
        - PEAK_POSITION_INITIAL_GUESS: List of initial positions (nm)
        - PEAK_WIDTH_INITIAL_GUESS: List of initial widths (nm)
        - PEAK_HEIGHT_INITIAL_GUESS: List of initial heights
        - PEAK_WIDTH_MAX: Maximum width constraint (optional)
        - PEAK_POSITION_TOLERANCE: Position tolerance (optional)
        - FIT_RANGE_NM: (min, max) wavelength range for fitting
        - OPTIMIZER: 'RAdam', 'Adam', or 'NAdam'
        - LEARNING_RATE: Learning rate
        - NUM_ITERATIONS: Number of optimization iterations
        - REG_*: Regularization weights
    
    spectra : np.ndarray, shape (N_batch, N_wl)
        Batch of spectra to fit
    wavelengths : np.ndarray, shape (N_wl,)
        Wavelength array
    use_gpu : bool
        Whether to use GPU
    
    Returns:
    --------
    Tuple of:
        - fitted_spectra: np.ndarray, shape (N_batch, N_wl)
        - params_list: List of N_batch dicts with fitted parameters
        - r2_scores: np.ndarray, shape (N_batch,)
    """
    fitter = LorentzianFitter(args, use_gpu)
    return fitter.fit(spectra, wavelengths)


def fit_fano_batch(args: Dict[str, Any],
                   spectra: np.ndarray,
                   wavelengths: np.ndarray,
                   use_gpu: bool = False) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
    """
    Fit Fano resonance model to multiple spectra using PyTorch batch processing
    
    Parameters:
    -----------
    args : Dict[str, Any]
        Configuration dictionary containing:
        - NUM_BRIGHT_MODES: Number of bright modes
        - NUM_DARK_MODES: Number of dark modes
        - BRIGHT_POSITION_INITIAL_GUESS: List of bright mode positions
        - BRIGHT_WIDTH_INITIAL_GUESS: List of bright mode widths
        - BRIGHT_HEIGHT_INITIAL_GUESS: List of bright mode heights
        - DARK_POSITION_INITIAL_GUESS: List of dark mode positions
        - DARK_WIDTH_INITIAL_GUESS: List of dark mode widths
        - DARK_HEIGHT_INITIAL_GUESS: List of dark mode heights
        - DARK_PHASE_INITIAL_GUESS: List of dark mode phases (optional)
        - BRIGHT/DARK_WIDTH_MAX: Width constraints
        - BRIGHT/DARK_POSITION_TOLERANCE: Position tolerances
        - FIT_RANGE_NM: (min, max) wavelength range
        - OPTIMIZER: Optimizer name
        - LEARNING_RATE_BRIGHT/DARK: Learning rates for each stage
        - NUM_ITERATIONS_BRIGHT/DARK: Iterations for each stage
    
    spectra : np.ndarray, shape (N_batch, N_wl)
    wavelengths : np.ndarray, shape (N_wl,)
    use_gpu : bool
    
    Returns:
    --------
    Same as fit_lorentz_batch
    """
    fitter = FanoFitter(args, use_gpu)
    return fitter.fit(spectra, wavelengths)


# =============================================================================
# UTILITY FUNCTIONS (Keep existing plot functions, etc.)
# =============================================================================

def plot_spectrum(wavelengths: np.ndarray,
                  spectrum: np.ndarray,
                  fit: np.ndarray,
                  title: str,
                  save_path: Path,
                  dpi: int = 300,
                  params: Optional[Dict[str, float]] = None,
                  snr: Optional[float] = None,
                  args: Optional[Dict[str, Any]] = None,
                  show_fit: bool = True):
    """
    Plot spectrum with fitted curve
    
    This function is kept unchanged for compatibility
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get output unit
    output_unit = args.get('OUTPUT_UNIT', 'eV') if args else 'eV'
    x_label = 'Energy (eV)' if output_unit == 'eV' else 'Wavelength (nm)'
    
    # Get fitting model
    fitting_model = args.get('FITTING_MODEL', 'lorentzian') if args else 'lorentzian'
    
    # Convert to eV if needed
    if output_unit == 'eV':
        x_plot = 1239.842 / wavelengths
        # Sort by increasing energy
        sort_idx = np.argsort(x_plot)
        x_plot = x_plot[sort_idx]
        spectrum = spectrum[sort_idx]
        if show_fit:
            fit = fit[sort_idx]
    else:
        x_plot = wavelengths
    
    # Plot
    ax.plot(x_plot, spectrum, 'o-', markersize=2, linewidth=1, label='Data', alpha=0.7)
    
    if show_fit:
        ax.plot(x_plot, fit, 'r-', linewidth=2, label='Fit')
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Intensity (a.u.)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add parameter text if provided
    if params is not None and show_fit:
        param_text = ""
        
        if fitting_model == 'lorentzian':
            num_peaks = sum(1 for key in params.keys() if key.startswith('b'))
            for i in range(1, num_peaks + 1):
                if f'b{i}' in params:
                    param_text += f"Peak {i}: λ={params[f'b{i}']:.1f} nm, FWHM={params[f'c{i}']:.1f} nm\n"
        
        elif fitting_model == 'fano':
            num_bright = sum(1 for key in params.keys() if 'bright' in key and 'lambda' in key)
            for i in range(1, num_bright + 1):
                if f'bright{i}_lambda' in params:
                    param_text += f"Bright {i}: λ={params[f'bright{i}_lambda']:.1f} nm, γ={params[f'bright{i}_gamma']:.1f} nm\n"
        
        if param_text:
            ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)