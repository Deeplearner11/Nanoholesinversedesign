import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import colour
from colour import MSDS_CMFS, SDS_ILLUMINANTS, SpectralDistribution
import argparse
import os
import random
from tqdm import tqdm
from scipy import integrate

# Set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class NanoholeDataset(Dataset):
    """Dataset class for nanohole data"""
    
    def __init__(self, structures, colors, transform=None):
        """
        Args:
            structures: numpy array of shape (N, 3) containing [diameter, pitch, depth]
            colors: numpy array of shape (N, 3) containing [x, y, Y] CIE coordinates
            transform: optional transform to be applied
        """
        self.structures = torch.FloatTensor(structures)
        self.colors = torch.FloatTensor(colors)
        self.transform = transform
        
    def __len__(self):
        return len(self.structures)
    
    def __getitem__(self, idx):
        structure = self.structures[idx]
        color = self.colors[idx]
        
        if self.transform:
            structure = self.transform(structure)
            color = self.transform(color)
            
        return structure, color

class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    
    def __init__(self, input_size, output_size, hidden_dim=64, num_layers=3):
        super(MLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
        
        layers.append(nn.Linear(hidden_dim, output_size))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class MLP_Sigmoid(nn.Module):
    """MLP with sigmoid output activation"""
    
    def __init__(self, input_size, output_size, hidden_dim=64, num_layers=3):
        super(MLP_Sigmoid, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
        
        layers.append(nn.Linear(hidden_dim, output_size))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class MLP_Inverse(nn.Module):
    """MLP for inverse model: color coordinates -> structural parameters"""
    
    def __init__(self, input_size=3, output_size=3, hidden_dim=128, num_layers=3, dropout_rate=0.2):
        super(MLP_Inverse, self).__init__()
        
        layers = []
        
        # First layer with larger hidden dimension for better capacity
        layers.append(nn.Linear(input_size, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer - NO activation function since data is already normalized
        layers.append(nn.Linear(hidden_dim, output_size))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class TandemNet(nn.Module):
    """Tandem Neural Network for inverse design"""
    
    def __init__(self, forward_model, inverse_model):
        super(TandemNet, self).__init__()
        self.forward_model = forward_model
        self.inverse_model = inverse_model
    
    def forward(self, structure, color_target):
        """
        Forward pass for training
        Args:
            structure: input structure parameters (not used during training)
            color_target: target color coordinates
        Returns:
            predicted_color: color coordinates from forward model
        """
        # Inverse model predicts structure from target color
        predicted_structure = self.inverse_model(color_target)
        # Forward model predicts color from predicted structure
        predicted_color = self.forward_model(predicted_structure)
        return predicted_color
    
    def predict_structure(self, color_target):
        """Predict structure parameters from target color"""
        return self.inverse_model(color_target)
# Color conversion matrices
M_RGB_TO_XYZ = np.array([
    [0.5767309, 0.1855540, 0.1881852],
    [0.2973769, 0.6273491, 0.0752741],
    [0.0270343, 0.0706872, 0.9911085]])

M_sRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]])

M_XYZ_TO_sRGB = np.linalg.inv(M_sRGB_TO_XYZ)    
M_XYZ_TO_RGB = np.linalg.inv(M_RGB_TO_XYZ)   


def XYZ_to_xyY(xyz_a):
    """Convert XYZ to xyY coordinates"""
    len_a = len(xyz_a)
    xyY_a = np.zeros([len_a, 3])
    temp_a = np.sum(xyz_a, axis=1)
    
    for i in range(len_a):
        if temp_a[i] == 0.0:
            xyY_a[i, 0] = 0.313  # Standard illuminant white point
            xyY_a[i, 1] = 0.329
            xyY_a[i, 2] = 0
            continue
        xyY_a[i, 0] = xyz_a[i, 0] / temp_a[i]  # x chromaticity
        xyY_a[i, 1] = xyz_a[i, 1] / temp_a[i]  # y chromaticity
        xyY_a[i, 2] = xyz_a[i, 1]              # Y luminance

    return xyY_a


def xyY_to_XYZ(xyY):
    """Convert xyY to XYZ coordinates"""
    # http://www.brucelindbloom.com/index.html?Eqn_xyY_to_XYZ.html
    temp = len(xyY)
    XYZ = np.zeros([temp, 3])
    
    for i in range(temp):
        if xyY[i, 1] == 0:
            continue
        XYZ[i, 0] = xyY[i, 0] * xyY[i, 2] / xyY[i, 1]  # X
        XYZ[i, 1] = xyY[i, 2]                           # Y
        XYZ[i, 2] = (1 - xyY[i, 0] - xyY[i, 1]) * xyY[i, 2] / xyY[i, 1]  # Z
    
    return XYZ


def xyY_to_RGB(xyY):
    """Convert xyY to RGB"""
    return np.dot(xyY_to_XYZ(xyY), np.transpose(M_XYZ_TO_RGB))


def RGB_to_xyY(RGB):
    """Convert RGB to xyY"""
    return XYZ_to_xyY(np.dot(RGB, np.transpose(M_RGB_TO_XYZ)))

def calculate_cie_coordinates(reflectance_data, wavelengths=None):
    """
    Calculate CIE coordinates from reflectance data using proper color science
    
    Args:
        reflectance_data: Array of reflectance spectra (n_samples x n_wavelengths)
        wavelengths: Wavelength array (default: 380-780nm at 5nm intervals)
    
    Returns:
        xyY coordinates (n_samples x 3) where columns are [x, y, Y]
    """
    
    if wavelengths is None:
        # Use standard range with 5nm intervals for better accuracy
        wavelengths = np.arange(380, 781, 5)  # 380-780nm at 5nm intervals
    
    # Get CIE 1931 2° Standard Observer color matching functions
    cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    
    # Get D65 illuminant
    illuminant_D65 = SDS_ILLUMINANTS['D65']
    
    # Ensure we have the right wavelength range
    wavelengths_nm = wavelengths
    
    # Sample the CMFs and illuminant at our wavelength points
    cmfs_values = cmfs.values
    illuminant_values = illuminant_D65.values
    
    # Get the wavelength ranges for interpolation
    cmfs_wavelengths = cmfs.wavelengths
    illuminant_wavelengths = illuminant_D65.wavelengths
    
    # Interpolate CMFs to our wavelength grid
    x_bar = np.interp(wavelengths_nm, cmfs_wavelengths, cmfs_values[:, 0])
    y_bar = np.interp(wavelengths_nm, cmfs_wavelengths, cmfs_values[:, 1])
    z_bar = np.interp(wavelengths_nm, cmfs_wavelengths, cmfs_values[:, 2])
    
    # Interpolate illuminant to our wavelength grid
    illuminant_interp = np.interp(wavelengths_nm, illuminant_wavelengths, illuminant_values)
    
    # Calculate normalization constant k
    # k = 100 / ∫ S(λ) * ȳ(λ) dλ
    k = 100.0 / np.trapz(illuminant_interp * y_bar, wavelengths_nm)
    
    # Calculate XYZ for each reflectance spectrum
    xyz_coords = []
    
    for reflectance in reflectance_data:
        # Ensure reflectance is the right length
        if len(reflectance) != len(wavelengths_nm):
            # Interpolate reflectance to match our wavelength grid
            # Assume input wavelengths are evenly spaced from 380-780nm
            input_wavelengths = np.linspace(380, 780, len(reflectance))
            reflectance_interp = np.interp(wavelengths_nm, input_wavelengths, reflectance)
        else:
            reflectance_interp = reflectance
        
        # Calculate stimulus (reflectance * illuminant)
        stimulus = reflectance_interp * illuminant_interp
        
        # Calculate XYZ using trapezoidal integration
        # X = k * ∫ R(λ) * S(λ) * x̄(λ) dλ
        # Y = k * ∫ R(λ) * S(λ) * ȳ(λ) dλ  
        # Z = k * ∫ R(λ) * S(λ) * z̄(λ) dλ
        
        X = k * np.trapz(stimulus * x_bar, wavelengths_nm)
        Y = k * np.trapz(stimulus * y_bar, wavelengths_nm)
        Z = k * np.trapz(stimulus * z_bar, wavelengths_nm)
        
        xyz_coords.append([X, Y, Z])
    
    xyz_coords = np.array(xyz_coords)
    
    # Convert XYZ to xyY using the existing function
    xyY_coords = XYZ_to_xyY(xyz_coords)
    
    return xyY_coords

def load_and_preprocess_data(csv_path, save_processed=True):
    """Load and preprocess nanohole data"""
    
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Extract structure parameters
    structures = df[['diameter', 'pitch', 'depth']].values
    
    # Extract reflectance data
    reflectance_columns = [col for col in df.columns if col.startswith('R_')]
    reflectance_columns.sort(key=lambda x: int(x.split('_')[1]))
    reflectance_data = df[reflectance_columns].values
    
    print("Calculating CIE coordinates...")
    colors = calculate_cie_coordinates(reflectance_data)
    
    # Create processed dataset with just input/output parameters
    if save_processed:
        processed_df = pd.DataFrame({
            'diameter': structures[:, 0],
            'pitch': structures[:, 1],
            'depth': structures[:, 2],
            'x': colors[:, 0],
            'y': colors[:, 1],
            'Y': colors[:, 2]
        })
        
        # Save the processed dataset
        processed_csv_path = 'nanohole_processed_data.csv'
        processed_df.to_csv(processed_csv_path, index=False)
        print(f"Processed dataset saved to: {processed_csv_path}")
        
        # Display first few rows
        print(f"\nProcessed dataset preview:")
        print(processed_df.head(10))
        
        # Save summary statistics
        summary_stats = processed_df.describe()
        summary_stats.to_csv('nanohole_data_statistics.csv')
        print(f"Summary statistics saved to: nanohole_data_statistics.csv")
    
    # Data statistics
    print(f"\nData Statistics:")
    print(f"Number of samples: {len(structures)}")
    print(f"Structure parameters shape: {structures.shape}")
    print(f"Color coordinates shape: {colors.shape}")
    
    print(f"\nStructure ranges:")
    print(f"Diameter: {structures[:, 0].min():.1f} - {structures[:, 0].max():.1f}")
    print(f"Pitch: {structures[:, 1].min():.1f} - {structures[:, 1].max():.1f}")
    print(f"Depth: {structures[:, 2].min():.1f} - {structures[:, 2].max():.1f}")
    
    print(f"\nColor ranges:")
    print(f"x: {colors[:, 0].min():.4f} - {colors[:, 0].max():.4f}")
    print(f"y: {colors[:, 1].min():.4f} - {colors[:, 1].max():.4f}")
    print(f"Y: {colors[:, 2].min():.4f} - {colors[:, 2].max():.4f}")
    
    return structures, colors

def normalize_data(structures, colors):
    """Normalize the data using MinMaxScaler"""
    
    # Normalize structures to [0, 1]
    structure_scaler = MinMaxScaler()
    structures_norm = structure_scaler.fit_transform(structures)
    
    # Normalize colors to [0, 1]
    color_scaler = MinMaxScaler()
    colors_norm = color_scaler.fit_transform(colors)
    
    # Save normalized data
    normalized_df = pd.DataFrame({
        'diameter_norm': structures_norm[:, 0],
        'pitch_norm': structures_norm[:, 1],
        'depth_norm': structures_norm[:, 2],
        'x_norm': colors_norm[:, 0],
        'y_norm': colors_norm[:, 1],
        'Y_norm': colors_norm[:, 2]
    })
    normalized_df.to_csv('normalized_nanohole_data.csv', index=False)
    print(f"Normalized dataset saved to: normalized_nanohole_data.csv")
    
    return structures_norm, colors_norm, structure_scaler, color_scaler

def create_data_loaders(structures, colors, batch_size=128, test_size=0.2, val_size=0.1):
    """Create train, validation, and test data loaders"""
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        structures, colors, test_size=test_size, random_state=42
    )
    
    # Second split: separate train and validation from remaining data
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42
    )
    
    print(f"\nDataset splits:")
    print(f"Train: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Create datasets
    train_dataset = NanoholeDataset(X_train, y_train)
    val_dataset = NanoholeDataset(X_val, y_val)
    test_dataset = NanoholeDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def create_data_visualization(structures, colors, save_path=None):
    """Create comprehensive data visualization"""
    
    # Create processed dataframe for plotting
    df_plot = pd.DataFrame({
        'diameter': structures[:, 0],
        'pitch': structures[:, 1], 
        'depth': structures[:, 2],
        'x': colors[:, 0],
        'y': colors[:, 1],
        'Y': colors[:, 2]
    })
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Nanohole Dataset Analysis', fontsize=16, fontweight='bold')
    
    # Input parameter distributions
    input_params = ['diameter', 'pitch', 'depth']
    for i, param in enumerate(input_params):
        axes[0, i].hist(df_plot[param], bins=50, alpha=0.7, color=f'C{i}')
        axes[0, i].set_title(f'{param.title()} Distribution')
        axes[0, i].set_xlabel(param.title())
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].grid(True, alpha=0.3)
    
    # Output parameter distributions
    output_params = ['x', 'y', 'Y']
    for i, param in enumerate(output_params):
        axes[1, i].hist(df_plot[param], bins=50, alpha=0.7, color=f'C{i+3}')
        axes[1, i].set_title(f'CIE {param} Distribution')
        axes[1, i].set_xlabel(f'CIE {param}')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].grid(True, alpha=0.3)
    
    # Correlation plots
    correlations = [
        ('diameter', 'x'),
        ('pitch', 'y'), 
        ('depth', 'Y')
    ]
    
    for i, (input_param, output_param) in enumerate(correlations):
        scatter = axes[2, i].scatter(df_plot[input_param], df_plot[output_param], 
                                   c=df_plot['Y'], cmap='viridis', alpha=0.6, s=20)
        axes[2, i].set_xlabel(input_param.title())
        axes[2, i].set_ylabel(f'CIE {output_param}')
        axes[2, i].set_title(f'{input_param.title()} vs CIE {output_param}')
        axes[2, i].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[2, i], label='CIE Y (Brightness)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print correlation matrix
    print("\nCorrelation Matrix:")
    correlation_matrix = df_plot.corr()
    print(correlation_matrix.round(3))
    
    return df_plot
    """Create train, validation, and test data loaders"""
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        structures, colors, test_size=test_size, random_state=42
    )
    
    # Second split: separate train and validation from remaining data
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42
    )
    
    print(f"\nDataset splits:")
    print(f"Train: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Create datasets
    train_dataset = NanoholeDataset(X_train, y_train)
    val_dataset = NanoholeDataset(X_val, y_val)
    test_dataset = NanoholeDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_forward_model(structures, colors, epochs=1000, batch_size=128):
    """Train the forward model (structure -> color)"""
    
    print("\n" + "="*50)
    print("Training Forward Model")
    print("="*50)
    
    # Normalize data
    structures_norm, colors_norm, struct_scaler, color_scaler = normalize_data(structures, colors)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        structures_norm, colors_norm, batch_size
    )
    
    # Create model
    forward_model = MLP(input_size=3, output_size=3, hidden_dim=128, num_layers=4).to(DEVICE)
    
    # Set up training
    optimizer = torch.optim.Adam(forward_model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(epochs), desc="Training Forward Model"):
        # Training
        forward_model.train()
        train_loss = 0
        for structures_batch, colors_batch in train_loader:
            structures_batch = structures_batch.to(DEVICE)
            colors_batch = colors_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = forward_model(structures_batch)
            loss = criterion(outputs, colors_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        forward_model.eval()
        val_loss = 0
        with torch.no_grad():
            for structures_batch, colors_batch in val_loader:
                structures_batch = structures_batch.to(DEVICE)
                colors_batch = colors_batch.to(DEVICE)
                
                outputs = forward_model(structures_batch)
                loss = criterion(outputs, colors_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': forward_model.state_dict(),
                'struct_scaler': struct_scaler,
                'color_scaler': color_scaler,
                'epoch': epoch,
                'val_loss': val_loss
            }, 'best_forward_model.pth')
        
        scheduler.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return forward_model, struct_scaler, color_scaler, train_losses, val_losses

import torch
import torch.nn as nn
from tqdm import tqdm

def train_tandem_model(structures, colors, forward_model, struct_scaler, color_scaler, 
                      epochs=2000, batch_size=128):
    """Train a tandem neural network to predict structures from colors, optimizing color reconstruction.
    
    Args:
        structures: Array of structures (e.g., nanohole diameter, pitch, depth).
        colors: Array of colors (e.g., RGB or spectral data).
        forward_model: Pre-trained forward model (structure -> color).
        struct_scaler: Scaler for structures.
        color_scaler: Scaler for colors.
        epochs: Number of training epochs.
        batch_size: Batch size for data loaders.
    
    Returns:
        tandem_model: Trained tandem model.
        train_losses: List of training losses (color-based).
        val_losses: List of validation tandem losses (color-based).
        val_losses_inverse: List of validation inverse losses (structure-based).
    """
    
    print("\n" + "="*50)
    print("Training Tandem Model for Structure Prediction")
    print("="*50)
    
    # Normalize data
    structures_norm = struct_scaler.transform(structures)
    colors_norm = color_scaler.transform(colors)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        structures_norm, colors_norm, batch_size
    )
    
    # Create inverse model (color -> structure) with sigmoid activation
    inverse_model = MLP_Sigmoid(input_size=3, output_size=3, hidden_dim=128, num_layers=4).to(DEVICE)
    
    # Create tandem model
    tandem_model = TandemNet(forward_model, inverse_model).to(DEVICE)
    
    # Freeze forward model parameters
    for param in tandem_model.forward_model.parameters():
        param.requires_grad = False
    
    # Set up training
    optimizer = torch.optim.Adam(tandem_model.inverse_model.parameters(), 
                                lr=1e-3, betas=(0.5, 0.999), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.5)
    criterion = nn.MSELoss()
    
    # Ensure forward model is in eval mode
    tandem_model.forward_model.eval()
    
    # Training loop
    train_losses = []
    val_losses = []
    val_losses_inverse = []
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(epochs), desc="Training Tandem Model"):
        # Training
        tandem_model.inverse_model.train()
        tandem_model.forward_model.eval()
        train_loss = 0
        for structures_batch, colors_batch in train_loader:
            structures_batch = structures_batch.to(DEVICE)
            colors_batch = colors_batch.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass through tandem network
            # Step 1: Predict structures from colors (inverse model)
            predicted_structures = tandem_model.inverse_model(colors_batch)
            
            # Step 2: Predict colors from predicted structures (forward model)
            reconstructed_colors = tandem_model.forward_model(predicted_structures)
            
            # Step 3: Compute loss between reconstructed and target colors
            loss = criterion(reconstructed_colors, colors_batch)
            # loss = criterion(predicted_structures, structures_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        tandem_model.eval()
        val_loss = 0
        val_loss_inverse = 0
        
        with torch.no_grad():
            for structures_batch, colors_batch in val_loader:
                structures_batch = structures_batch.to(DEVICE)
                colors_batch = colors_batch.to(DEVICE)
                
                # Tandem network loss (color reconstruction)
                predicted_structures = tandem_model.inverse_model(colors_batch)
                reconstructed_colors = tandem_model.forward_model(predicted_structures)
                tandem_loss = criterion(reconstructed_colors, colors_batch)
                val_loss += tandem_loss.item()
                
                # Inverse model loss (structure prediction accuracy)
                inverse_loss = criterion(predicted_structures, structures_batch)
                val_loss_inverse += inverse_loss.item()
        
        val_loss /= len(val_loader)
        val_loss_inverse /= len(val_loader)
        val_losses.append(val_loss)
        val_losses_inverse.append(val_loss_inverse)
        
        # Save best model based on validation tandem loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': tandem_model.state_dict(),
                'inverse_model_state_dict': tandem_model.inverse_model.state_dict(),
                'forward_model_state_dict': forward_model.state_dict(),
                'struct_scaler': struct_scaler,
                'color_scaler': color_scaler,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_loss_inverse': val_loss_inverse
            }, 'best_tandem_model.pth')
        
        scheduler.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Tandem Loss: {val_loss:.6f}, Val Structure Loss: {val_loss_inverse:.6f}")
    
    return tandem_model, train_losses, val_losses, val_losses_inverse

def plot_training_history(forward_losses, tandem_losses, save_path=None):
    """Plot training history"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Forward model losses
    axes[0, 0].plot(forward_losses[0], label='Train Loss', color='blue')
    axes[0, 0].plot(forward_losses[1], label='Val Loss', color='orange')
    axes[0, 0].set_title('Forward Model Training')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Tandem model losses
    axes[0, 1].plot(tandem_losses[0], label='Train Loss', color='blue')
    axes[0, 1].plot(tandem_losses[1], label='Val Loss', color='orange')
    axes[0, 1].plot(tandem_losses[2], label='Val Inverse Loss', color='red')
    axes[0, 1].set_title('Tandem Model Training')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Log scale plots
    axes[1, 0].semilogy(forward_losses[0], label='Train Loss', color='blue')
    axes[1, 0].semilogy(forward_losses[1], label='Val Loss', color='orange')
    axes[1, 0].set_title('Forward Model Training (Log Scale)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MSE Loss (log)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].semilogy(tandem_losses[0], label='Train Loss', color='blue')
    axes[1, 1].semilogy(tandem_losses[1], label='Val Loss', color='orange')
    axes[1, 1].semilogy(tandem_losses[2], label='Val Inverse Loss', color='red')
    axes[1, 1].set_title('Tandem Model Training (Log Scale)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MSE Loss (log)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def evaluate_model(tandem_model, structures, colors, struct_scaler, color_scaler, 
                  n_samples=200):
    """Evaluate the trained tandem model"""
    
    print("\n" + "="*50)
    print("Model Evaluation")
    print("="*50)
    
    # Normalize data
    structures_norm = struct_scaler.transform(structures)
    colors_norm = color_scaler.transform(colors)
    
    # Select random samples for evaluation
    indices = np.random.choice(len(structures), n_samples, replace=False)
    
    tandem_model.eval()
    
    results = []
    
    with torch.no_grad():
        for i in indices:
            # Original data
            orig_struct = structures[i]
            orig_color = colors[i]
            
            # Normalized inputs
            target_color = torch.FloatTensor(colors_norm[i]).unsqueeze(0).to(DEVICE)
            
            # Predict structure from color
            pred_struct_norm = tandem_model.predict_structure(target_color)
            pred_struct = struct_scaler.inverse_transform(pred_struct_norm.cpu().numpy())
            
            # Predict color from predicted structure
            pred_color_norm = tandem_model.forward_model(pred_struct_norm)
            pred_color = color_scaler.inverse_transform(pred_color_norm.cpu().numpy())
            
            results.append({
                'orig_diameter': orig_struct[0],
                'orig_pitch': orig_struct[1],
                'orig_depth': orig_struct[2],
                'pred_diameter': pred_struct[0, 0],
                'pred_pitch': pred_struct[0, 1],
                'pred_depth': pred_struct[0, 2],
                'orig_x': orig_color[0],
                'orig_y': orig_color[1],
                'orig_Y': orig_color[2],
                'pred_x': pred_color[0, 0],
                'pred_y': pred_color[0, 1],
                'pred_Y': pred_color[0, 2],
            })
    
    # Create evaluation DataFrame
    eval_df = pd.DataFrame(results)
    
    # Calculate Mean Absolute Errors
    struct_mae = {
        'diameter_mae': np.abs(eval_df['orig_diameter'] - eval_df['pred_diameter']).mean(),
        'pitch_mae': np.abs(eval_df['orig_pitch'] - eval_df['pred_pitch']).mean(),
        'depth_mae': np.abs(eval_df['orig_depth'] - eval_df['pred_depth']).mean(),
    }
    
    color_mae = {
        'x_mae': np.abs(eval_df['orig_x'] - eval_df['pred_x']).mean(),
        'y_mae': np.abs(eval_df['orig_y'] - eval_df['pred_y']).mean(),
        'Y_mae': np.abs(eval_df['orig_Y'] - eval_df['pred_Y']).mean(),
    }
    
    # Calculate R² values
    def calculate_r2(y_true, y_pred):
        """Calculate R² (coefficient of determination)"""
        ss_res = np.sum((y_true - y_pred) ** 2)  # Sum of squares of residuals
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    struct_r2 = {
        'diameter_r2': calculate_r2(eval_df['orig_diameter'], eval_df['pred_diameter']),
        'pitch_r2': calculate_r2(eval_df['orig_pitch'], eval_df['pred_pitch']),
        'depth_r2': calculate_r2(eval_df['orig_depth'], eval_df['pred_depth']),
    }
    
    color_r2 = {
        'x_r2': calculate_r2(eval_df['orig_x'], eval_df['pred_x']),
        'y_r2': calculate_r2(eval_df['orig_y'], eval_df['pred_y']),
        'Y_r2': calculate_r2(eval_df['orig_Y'], eval_df['pred_Y']),
    }
    
    # Calculate Root Mean Square Errors for additional context
    struct_rmse = {
        'diameter_rmse': np.sqrt(np.mean((eval_df['orig_diameter'] - eval_df['pred_diameter']) ** 2)),
        'pitch_rmse': np.sqrt(np.mean((eval_df['orig_pitch'] - eval_df['pred_pitch']) ** 2)),
        'depth_rmse': np.sqrt(np.mean((eval_df['orig_depth'] - eval_df['pred_depth']) ** 2)),
    }
    
    color_rmse = {
        'x_rmse': np.sqrt(np.mean((eval_df['orig_x'] - eval_df['pred_x']) ** 2)),
        'y_rmse': np.sqrt(np.mean((eval_df['orig_y'] - eval_df['pred_y']) ** 2)),
        'Y_rmse': np.sqrt(np.mean((eval_df['orig_Y'] - eval_df['pred_Y']) ** 2)),
    }
    
    # Print results
    print("Structure Prediction Metrics:")
    print("  Mean Absolute Error (MAE):")
    for param, error in struct_mae.items():
        print(f"    {param}: {error:.2f}")
    print("  Root Mean Square Error (RMSE):")
    for param, error in struct_rmse.items():
        print(f"    {param}: {error:.2f}")
    print("  R² Score:")
    for param, r2 in struct_r2.items():
        print(f"    {param}: {r2:.4f}")
    
    print("\nColor Reconstruction Metrics:")
    print("  Mean Absolute Error (MAE):")
    for param, error in color_mae.items():
        print(f"    {param}: {error:.4f}")
    print("  Root Mean Square Error (RMSE):")
    for param, error in color_rmse.items():
        print(f"    {param}: {error:.4f}")
    print("  R² Score:")
    for param, r2 in color_r2.items():
        print(f"    {param}: {r2:.4f}")
    
    # Overall metrics
    overall_struct_r2 = np.mean(list(struct_r2.values()))
    overall_color_r2 = np.mean(list(color_r2.values()))
    
    print(f"\nOverall Performance:")
    print(f"  Structure Prediction R²: {overall_struct_r2:.4f}")
    print(f"  Color Reconstruction R²: {overall_color_r2:.4f}")
    print(f"  Combined R²: {(overall_struct_r2 + overall_color_r2) / 2:.4f}")
    
    # Plot evaluation results with R² values in titles
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Structure parameters
    struct_params = ['diameter', 'pitch', 'depth']
    for i, param in enumerate(struct_params):
        orig_col = f'orig_{param}'
        pred_col = f'pred_{param}'
        r2_val = struct_r2[f'{param}_r2']
        mae_val = struct_mae[f'{param}_mae']
        
        axes[0, i].scatter(eval_df[orig_col], eval_df[pred_col], alpha=0.7)
        min_val = min(eval_df[orig_col].min(), eval_df[pred_col].min())
        max_val = max(eval_df[orig_col].max(), eval_df[pred_col].max())
        axes[0, i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect fit')
        axes[0, i].set_xlabel(f'Original {param.title()}')
        axes[0, i].set_ylabel(f'Predicted {param.title()}')
        axes[0, i].set_title(f'{param.title()} Prediction\nR² = {r2_val:.3f}, MAE = {mae_val:.2f}')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].legend()
    
    # Color parameters
    color_params = ['x', 'y', 'Y']
    for i, param in enumerate(color_params):
        orig_col = f'orig_{param}'
        pred_col = f'pred_{param}'
        r2_val = color_r2[f'{param}_r2']
        mae_val = color_mae[f'{param}_mae']
        
        axes[1, i].scatter(eval_df[orig_col], eval_df[pred_col], alpha=0.7)
        min_val = min(eval_df[orig_col].min(), eval_df[pred_col].min())
        max_val = max(eval_df[orig_col].max(), eval_df[pred_col].max())
        axes[1, i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect fit')
        axes[1, i].set_xlabel(f'Original {param}')
        axes[1, i].set_ylabel(f'Predicted {param}')
        axes[1, i].set_title(f'{param} Reconstruction\nR² = {r2_val:.3f}, MAE = {mae_val:.4f}')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].legend()
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a comprehensive metrics dictionary for easy access
    metrics = {
        'structure_mae': struct_mae,
        'structure_rmse': struct_rmse,
        'structure_r2': struct_r2,
        'color_mae': color_mae,
        'color_rmse': color_rmse,
        'color_r2': color_r2,
        'overall_structure_r2': overall_struct_r2,
        'overall_color_r2': overall_color_r2,
        'overall_combined_r2': (overall_struct_r2 + overall_color_r2) / 2
    }
    
    return eval_df, metrics

def main():
    """Main training function"""
    
    # Load and preprocess data (saves processed CSV automatically)
    structures, colors = load_and_preprocess_data('combined_nanohole_data70.csv', save_processed=True)
    
    # Create data visualization
    print("\nCreating data visualization...")
    df_processed = create_data_visualization(structures, colors, 'data_analysis.png')
    
    # Train forward model
    forward_model, struct_scaler, color_scaler, forward_train_losses, forward_val_losses = train_forward_model(
        structures, colors, epochs=1000
    )
    
    # Train tandem model
    tandem_model, tandem_train_losses, tandem_val_losses, tandem_val_inverse_losses = train_tandem_model(
        structures, colors, forward_model, struct_scaler, color_scaler, epochs=2000
    )
    
    # Plot training history
    plot_training_history(
        (forward_train_losses, forward_val_losses),
        (tandem_train_losses, tandem_val_losses, tandem_val_inverse_losses),
        'training_history.png'
    )
    
    # Evaluate model - NOW RETURNS TWO VALUES
    eval_results, metrics = evaluate_model(tandem_model, structures, colors, struct_scaler, color_scaler)
    
    # Save evaluation results
    eval_results.to_csv('model_evaluation_results.csv', index=False)
    
    # Save comprehensive metrics to a separate file
    import json
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary of key metrics
    print(f"\n" + "="*50)
    print("FINAL MODEL PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Overall Structure Prediction R²: {metrics['overall_structure_r2']:.4f}")
    print(f"Overall Color Reconstruction R²: {metrics['overall_color_r2']:.4f}")
    print(f"Combined Model Performance R²: {metrics['overall_combined_r2']:.4f}")
    
    print("\nTraining completed successfully!")
    print("Saved files:")
    print("  - nanohole_processed_data.csv (Main dataset: diameter, pitch, depth, x, y, Y)")
    print("  - nanohole_data_statistics.csv (Summary statistics)")
    print("  - best_forward_model.pth (Trained forward model)")
    print("  - best_tandem_model.pth (Trained tandem model)")
    print("  - data_analysis.png (Data visualization)")
    print("  - training_history.png (Training curves)")
    print("  - model_evaluation.png (Model performance)")
    print("  - model_evaluation_results.csv (Detailed evaluation results)")
    print("  - model_metrics.json (Comprehensive performance metrics)")
    
    return df_processed, tandem_model, struct_scaler, color_scaler, metrics

if __name__ == "__main__":
    main()