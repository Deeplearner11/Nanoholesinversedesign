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
    """Dataset class for nanohole data - Color to Structure mapping"""
    
    def __init__(self, colors, structures, transform=None):
        """
        Args:
            colors: numpy array of shape (N, 3) containing [x, y, Y] CIE coordinates (INPUTS)
            structures: numpy array of shape (N, 3) containing [diameter, pitch, depth] (OUTPUTS)
            transform: optional transform to be applied
        """
        self.colors = torch.FloatTensor(colors)
        self.structures = torch.FloatTensor(structures)
        self.transform = transform
        
    def __len__(self):
        return len(self.colors)
    
    def __getitem__(self, idx):
        color = self.colors[idx]
        structure = self.structures[idx]
        
        if self.transform:
            color = self.transform(color)
            structure = self.transform(structure)
            
        return color, structure

class MLP(nn.Module):
    """Multi-Layer Perceptron for Color to Structure mapping"""
    
    def __init__(self, input_size=3, output_size=3, hidden_dim=128, num_layers=4, dropout_rate=0.1):
        super(MLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Dropout(dropout_rate))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer - no activation since data is normalized
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
    """MLP for inverse model: structure parameters -> color coordinates"""
    
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
    """Tandem Neural Network for inverse design - validates structure predictions"""
    
    def __init__(self, main_model, validation_model):
        super(TandemNet, self).__init__()
        self.main_model = main_model  # Color -> Structure
        self.validation_model = validation_model  # Structure -> Color
    
    def forward(self, color_input, structure_target):
        """
        Forward pass for training
        Args:
            color_input: input color coordinates
            structure_target: target structure parameters (not used during training)
        Returns:
            predicted_color: color coordinates from validation model
        """
        # Main model predicts structure from color
        predicted_structure = self.main_model(color_input)
        # Validation model predicts color from predicted structure
        predicted_color = self.validation_model(predicted_structure)
        return predicted_color
    
    def predict_structure(self, color_input):
        """Predict structure parameters from color coordinates"""
        return self.main_model(color_input)

# Color conversion matrices (unchanged)
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

# Color conversion functions (unchanged)
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
    """Calculate CIE coordinates from reflectance data using proper color science"""
    
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
        X = k * np.trapz(stimulus * x_bar, wavelengths_nm)
        Y = k * np.trapz(stimulus * y_bar, wavelengths_nm)
        Z = k * np.trapz(stimulus * z_bar, wavelengths_nm)
        
        xyz_coords.append([X, Y, Z])
    
    xyz_coords = np.array(xyz_coords)
    
    # Convert XYZ to xyY using the existing function
    xyY_coords = XYZ_to_xyY(xyz_coords)
    
    return xyY_coords

def load_and_preprocess_data(csv_path, save_processed=True):
    """Load and preprocess nanohole data with color as input, structure as output"""
    
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Extract structure parameters (OUTPUTS)
    structures = df[['diameter', 'pitch', 'depth']].values
    
    # Extract reflectance data
    reflectance_columns = [col for col in df.columns if col.startswith('R_')]
    reflectance_columns.sort(key=lambda x: int(x.split('_')[1]))
    reflectance_data = df[reflectance_columns].values
    
    print("Calculating CIE coordinates...")
    colors = calculate_cie_coordinates(reflectance_data)
    # colors = df[['x', 'y', 'Y']].values  # Assuming these columns exist in the CSV
    
    # Create processed dataset with swapped input/output roles
    if save_processed:
        processed_df = pd.DataFrame({
            'x': colors[:, 0],      # INPUT: CIE x chromaticity
            'y': colors[:, 1],      # INPUT: CIE y chromaticity  
            'Y': colors[:, 2],      # INPUT: CIE Y luminance
            'diameter': structures[:, 0],  # OUTPUT: diameter
            'pitch': structures[:, 1],     # OUTPUT: pitch
            'depth': structures[:, 2]      # OUTPUT: depth
        })
        
        # Save the processed dataset
        processed_csv_path = 'nanohole_processed_data_color_to_structure.csv'
        processed_df.to_csv(processed_csv_path, index=False)
        print(f"Processed dataset saved to: {processed_csv_path}")
        
        # Display first few rows
        print(f"\nProcessed dataset preview (Color->Structure mapping):")
        print(processed_df.head(10))
        
        # Save summary statistics
        summary_stats = processed_df.describe()
        summary_stats.to_csv('nanohole_data_statistics_color_to_structure.csv')
        print(f"Summary statistics saved to: nanohole_data_statistics_color_to_structure.csv")
    
    # Data statistics
    print(f"\nData Statistics:")
    print(f"Number of samples: {len(colors)}")
    print(f"Color coordinates shape (INPUTS): {colors.shape}")
    print(f"Structure parameters shape (OUTPUTS): {structures.shape}")
    
    print(f"\nColor ranges (INPUTS):")
    print(f"x: {colors[:, 0].min():.4f} - {colors[:, 0].max():.4f}")
    print(f"y: {colors[:, 1].min():.4f} - {colors[:, 1].max():.4f}")
    print(f"Y: {colors[:, 2].min():.4f} - {colors[:, 2].max():.4f}")
    
    print(f"\nStructure ranges (OUTPUTS):")
    print(f"Diameter: {structures[:, 0].min():.1f} - {structures[:, 0].max():.1f}")
    print(f"Pitch: {structures[:, 1].min():.1f} - {structures[:, 1].max():.1f}")
    print(f"Depth: {structures[:, 2].min():.1f} - {structures[:, 2].max():.1f}")
    
    return colors, structures  # Note: swapped order - colors first (inputs), structures second (outputs)

def normalize_data(colors, structures):
    """Normalize the data using MinMaxScaler - colors as inputs, structures as outputs"""
    
    # Normalize colors (inputs) to [0, 1]
    color_scaler = MinMaxScaler()
    colors_norm = color_scaler.fit_transform(colors)
    
    # Normalize structures (outputs) to [0, 1]
    structure_scaler = MinMaxScaler()
    structures_norm = structure_scaler.fit_transform(structures)
    
    # Save normalized data
    normalized_df = pd.DataFrame({
        'x_norm': colors_norm[:, 0],
        'y_norm': colors_norm[:, 1],
        'Y_norm': colors_norm[:, 2],
        'diameter_norm': structures_norm[:, 0],
        'pitch_norm': structures_norm[:, 1],
        'depth_norm': structures_norm[:, 2]
    })
    normalized_df.to_csv('normalized_nanohole_data_color_to_structure.csv', index=False)
    print(f"Normalized dataset saved to: normalized_nanohole_data_color_to_structure.csv")
    
    return colors_norm, structures_norm, color_scaler, structure_scaler

def create_data_loaders(colors, structures, batch_size=128, test_size=0.2, val_size=0.1):
    """Create train, validation, and test data loaders - colors as inputs, structures as outputs"""
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        colors, structures, test_size=test_size, random_state=42
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
    
    # Create datasets (colors as inputs, structures as outputs)
    train_dataset = NanoholeDataset(X_train, y_train)
    val_dataset = NanoholeDataset(X_val, y_val)
    test_dataset = NanoholeDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def create_data_visualization(colors, structures, save_path=None):
    """Create comprehensive data visualization - colors as inputs, structures as outputs"""
    
    # Create processed dataframe for plotting
    df_plot = pd.DataFrame({
        'x': colors[:, 0],          # INPUT
        'y': colors[:, 1],          # INPUT
        'Y': colors[:, 2],          # INPUT
        'diameter': structures[:, 0],  # OUTPUT
        'pitch': structures[:, 1],     # OUTPUT
        'depth': structures[:, 2]      # OUTPUT
    })
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Nanohole Dataset Analysis (Color → Structure Mapping)', fontsize=16, fontweight='bold')
    
    # Input parameter distributions (Colors)
    input_params = ['x', 'y', 'Y']
    for i, param in enumerate(input_params):
        axes[0, i].hist(df_plot[param], bins=50, alpha=0.7, color=f'C{i}')
        axes[0, i].set_title(f'CIE {param} Distribution (INPUT)')
        axes[0, i].set_xlabel(f'CIE {param}')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].grid(True, alpha=0.3)
    
    # Output parameter distributions (Structures)
    output_params = ['diameter', 'pitch', 'depth']
    for i, param in enumerate(output_params):
        axes[1, i].hist(df_plot[param], bins=50, alpha=0.7, color=f'C{i+3}')
        axes[1, i].set_title(f'{param.title()} Distribution (OUTPUT)')
        axes[1, i].set_xlabel(param.title())
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].grid(True, alpha=0.3)
    
    # Correlation plots (Input vs Output)
    correlations = [
        ('x', 'diameter'),
        ('y', 'pitch'), 
        ('Y', 'depth')
    ]
    
    for i, (input_param, output_param) in enumerate(correlations):
        scatter = axes[2, i].scatter(df_plot[input_param], df_plot[output_param], 
                                   c=df_plot['Y'], cmap='viridis', alpha=0.6, s=20)
        axes[2, i].set_xlabel(f'CIE {input_param} (INPUT)')
        axes[2, i].set_ylabel(f'{output_param.title()} (OUTPUT)')
        axes[2, i].set_title(f'CIE {input_param} vs {output_param.title()}')
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

def train_main_model(colors, structures, epochs=2000, batch_size=128):
    """Train the main model (color -> structure)"""
    
    print("\n" + "="*50)
    print("Training Main Model (Color → Structure)")
    print("="*50)
    
    # Normalize data
    colors_norm, structures_norm, color_scaler, structure_scaler = normalize_data(colors, structures)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        colors_norm, structures_norm, batch_size
    )
    
    # Create model
    main_model = MLP(input_size=3, output_size=3, hidden_dim=64, num_layers=3).to(DEVICE)
    
    # Set up training
    optimizer = torch.optim.Adam(main_model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(epochs), desc="Training Main Model"):
        # Training
        main_model.train()
        train_loss = 0
        for colors_batch, structures_batch in train_loader:
            colors_batch = colors_batch.to(DEVICE)
            structures_batch = structures_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = main_model(colors_batch)
            loss = criterion(outputs, structures_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        main_model.eval()
        val_loss = 0
        with torch.no_grad():
            for colors_batch, structures_batch in val_loader:
                colors_batch = colors_batch.to(DEVICE)
                structures_batch = structures_batch.to(DEVICE)
                
                outputs = main_model(colors_batch)
                loss = criterion(outputs, structures_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': main_model.state_dict(),
                'color_scaler': color_scaler,
                'structure_scaler': structure_scaler,
                'epoch': epoch,
                'val_loss': val_loss
            }, 'best_main_model_color_to_structure.pth')
        
        scheduler.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return main_model, color_scaler, structure_scaler, train_losses, val_losses



def train_inverse_model(colors, structures, epochs=1000, batch_size=128):
    """Train the inverse model (structure -> color) separately first"""
    
    print("\n" + "="*50)
    print("Training Inverse Model (Structure → Color)")
    print("="*50)
    
    # Normalize data
    colors_norm, structures_norm, color_scaler, structure_scaler = normalize_data(colors, structures)
    
    # Create data loaders (note: we swap the inputs/outputs for inverse model)
    train_loader, val_loader, test_loader = create_data_loaders(
        structures_norm, colors_norm, batch_size  # structures as input, colors as output
    )
    
    # Create inverse model
    inverse_model = MLP_Inverse(input_size=3, output_size=3, hidden_dim=128, num_layers=3).to(DEVICE)
    
    # Set up training
    optimizer = torch.optim.Adam(inverse_model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(epochs), desc="Training Inverse Model"):
        # Training
        inverse_model.train()
        train_loss = 0
        for structures_batch, colors_batch in train_loader:  # Note the order swap
            structures_batch = structures_batch.to(DEVICE)
            colors_batch = colors_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = inverse_model(structures_batch)  # structure -> color
            loss = criterion(outputs, colors_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        inverse_model.eval()
        val_loss = 0
        with torch.no_grad():
            for structures_batch, colors_batch in val_loader:
                structures_batch = structures_batch.to(DEVICE)
                colors_batch = colors_batch.to(DEVICE)
                
                outputs = inverse_model(structures_batch)
                loss = criterion(outputs, colors_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': inverse_model.state_dict(),
                'color_scaler': color_scaler,
                'structure_scaler': structure_scaler,
                'epoch': epoch,
                'val_loss': val_loss
            }, 'best_inverse_model_structure_to_color.pth')
        
        scheduler.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return inverse_model, color_scaler, structure_scaler, train_losses, val_losses


def train_tandem_model(colors, structures, main_model, pretrained_inverse_model, 
                                     color_scaler, structure_scaler, epochs=1000, batch_size=128):
    """Train tandem model with both main and inverse models pre-trained"""
    
    print("\n" + "="*50)
    print("Training Tandem Model with Pre-trained Components")
    print("="*50)
    
    # Normalize data
    colors_norm = color_scaler.transform(colors)
    structures_norm = structure_scaler.transform(structures)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        colors_norm, structures_norm, batch_size
    )
    
    # Create tandem model with pre-trained components
    tandem_model = TandemNet(main_model, pretrained_inverse_model).to(DEVICE)
    
    # Freeze main model parameters (keep main model fixed)
    for param in tandem_model.main_model.parameters():
        param.requires_grad = False
    
    # Allow inverse model parameters to be updated (fine-tune for consistency)
    for param in tandem_model.validation_model.parameters():
        param.requires_grad = True
    
    # Set up training - only for the inverse model parameters
    optimizer = torch.optim.Adam(
        tandem_model.validation_model.parameters(), 
        lr=1e-3,  # Lower learning rate since we're fine-tuning
        weight_decay=1e-5
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.7)
    criterion = nn.MSELoss()
    
    # Training loop
    train_losses_total = []
    train_losses_structure = []
    train_losses_color = []
    val_losses_total = []
    val_losses_structure = []
    val_losses_color = []
    best_val_loss = float('inf')
    
    # Loss weights - you can experiment with these
    structure_weight = 1.0    # Primary: structure prediction accuracy
    color_weight = 0        # Secondary: color reconstruction consistency
    
    for epoch in tqdm(range(epochs), desc="Training Tandem Model"):
        # Training
        tandem_model.train()
        train_loss_total = 0
        train_loss_struct = 0
        train_loss_col = 0
        
        for colors_batch, structures_batch in train_loader:
            colors_batch = colors_batch.to(DEVICE)
            structures_batch = structures_batch.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass through tandem network
            # Step 1: Predict structures from colors (frozen main model)
            with torch.no_grad():  # Main model is frozen, no gradients needed
                predicted_structures = tandem_model.main_model(colors_batch)
            
            # Step 2: Predict colors from predicted structures (trainable inverse model)
            reconstructed_colors = tandem_model.validation_model(predicted_structures)
            
            # Step 3: Compute losses
            # Primary loss: How well do we predict structures? (This is fixed since main model is frozen)
            structure_loss = criterion(predicted_structures, structures_batch)
            
            # Secondary loss: Consistency check - can we reconstruct input colors?
            color_consistency_loss = criterion(reconstructed_colors, colors_batch)
            
            # Since main model is frozen, we only optimize the consistency loss
            # But we still track structure loss for monitoring
            total_loss = color_consistency_loss  # Only this affects gradients
            
            total_loss.backward()
            optimizer.step()
            
            train_loss_total += total_loss.item()
            train_loss_struct += structure_loss.item()
            train_loss_col += color_consistency_loss.item()
        
        train_loss_total /= len(train_loader)
        train_loss_struct /= len(train_loader)
        train_loss_col /= len(train_loader)
        
        train_losses_total.append(train_loss_total)
        train_losses_structure.append(train_loss_struct)
        train_losses_color.append(train_loss_col)
        
        # Validation
        tandem_model.eval()
        val_loss_total = 0
        val_loss_struct = 0
        val_loss_col = 0
        
        with torch.no_grad():
            for colors_batch, structures_batch in val_loader:
                colors_batch = colors_batch.to(DEVICE)
                structures_batch = structures_batch.to(DEVICE)
                
                # Forward pass
                predicted_structures = tandem_model.main_model(colors_batch)
                reconstructed_colors = tandem_model.validation_model(predicted_structures)
                
                # Compute losses
                structure_loss = criterion(predicted_structures, structures_batch)
                color_consistency_loss = criterion(reconstructed_colors, colors_batch)
                total_loss = color_consistency_loss
                
                val_loss_total += total_loss.item()
                val_loss_struct += structure_loss.item()
                val_loss_col += color_consistency_loss.item()
        
        val_loss_total /= len(val_loader)
        val_loss_struct /= len(val_loader)
        val_loss_col /= len(val_loader)
        
        val_losses_total.append(val_loss_total)
        val_losses_structure.append(val_loss_struct)
        val_losses_color.append(val_loss_col)
        
        # Save best model based on overall consistency
        # You could also use val_loss_struct if you want to prioritize structure prediction
        if val_loss_total < best_val_loss:
            best_val_loss = val_loss_total
            torch.save({
                'tandem_model_state_dict': tandem_model.state_dict(),
                'main_model_state_dict': tandem_model.main_model.state_dict(),
                'validation_model_state_dict': tandem_model.validation_model.state_dict(),
                'color_scaler': color_scaler,
                'structure_scaler': structure_scaler,
                'epoch': epoch,
                'val_loss_total': val_loss_total,
                'val_loss_structure': val_loss_struct,
                'val_loss_color': val_loss_col,
                'structure_weight': structure_weight,
                'color_weight': color_weight
            }, 'best_tandem_model_pretrained.pth')
        
        scheduler.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}")
            print(f"  Train - Total: {train_loss_total:.6f}, Structure: {train_loss_struct:.6f}, Color: {train_loss_col:.6f}")
            print(f"  Val   - Total: {val_loss_total:.6f}, Structure: {val_loss_struct:.6f}, Color: {val_loss_col:.6f}")
    
    # Return organized losses
    train_losses = {
        'total': train_losses_total,
        'structure': train_losses_structure,
        'color': train_losses_color
    }
    
    val_losses = {
        'total': val_losses_total,
        'structure': val_losses_structure,
        'color': val_losses_color
    }
    
    return tandem_model, train_losses, val_losses
    
def plot_training_history(main_losses, tandem_losses, save_path=None):
    """Plot training history with corrected tandem loss structure"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Main model losses
    axes[0, 0].plot(main_losses[0], label='Train Loss', color='blue')
    axes[0, 0].plot(main_losses[1], label='Val Loss', color='orange')
    axes[0, 0].set_title('Main Model Training (Color → Structure)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Tandem model - total losses
    axes[0, 1].plot(tandem_losses['train']['total'], label='Train Total', color='blue')
    axes[0, 1].plot(tandem_losses['val']['total'], label='Val Total', color='orange')
    axes[0, 1].set_title('Tandem Model - Total Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Tandem model - structure losses (primary objective)
    axes[0, 2].plot(tandem_losses['train']['structure'], label='Train Structure', color='red')
    axes[0, 2].plot(tandem_losses['val']['structure'], label='Val Structure', color='darkred')
    axes[0, 2].set_title('Tandem Model - Structure Loss (Primary)')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('MSE Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Log scale plots
    axes[1, 0].semilogy(main_losses[0], label='Train Loss', color='blue')
    axes[1, 0].semilogy(main_losses[1], label='Val Loss', color='orange')
    axes[1, 0].set_title('Main Model Training (Log Scale)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MSE Loss (log)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Tandem model - color consistency losses
    axes[1, 1].plot(tandem_losses['train']['color'], label='Train Color', color='green')
    axes[1, 1].plot(tandem_losses['val']['color'], label='Val Color', color='darkgreen')
    axes[1, 1].set_title('Tandem Model - Color Consistency Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MSE Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Combined log scale plot for tandem model
    axes[1, 2].semilogy(tandem_losses['train']['total'], label='Train Total', color='blue')
    axes[1, 2].semilogy(tandem_losses['val']['total'], label='Val Total', color='orange')
    axes[1, 2].semilogy(tandem_losses['train']['structure'], label='Train Structure', color='red')
    axes[1, 2].semilogy(tandem_losses['val']['structure'], label='Val Structure', color='darkred')
    axes[1, 2].set_title('Tandem Model - All Losses (Log Scale)')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('MSE Loss (log)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def evaluate_model(tandem_model, colors, structures, color_scaler, structure_scaler, 
                  n_samples=200):
    """Evaluate the trained tandem model - colors as inputs, structures as outputs"""
    
    print("\n" + "="*50)
    print("Model Evaluation (Color → Structure)")
    print("="*50)
    
    # Normalize data
    colors_norm = color_scaler.transform(colors)
    structures_norm = structure_scaler.transform(structures)
    
    # Select random samples for evaluation
    indices = np.random.choice(len(colors), n_samples, replace=False)
    
    tandem_model.eval()
    
    results = []
    
    with torch.no_grad():
        for i in indices:
            # Original data
            orig_color = colors[i]
            orig_struct = structures[i]
            
            # Normalized inputs
            input_color = torch.FloatTensor(colors_norm[i]).unsqueeze(0).to(DEVICE)
            
            # Predict structure from color
            pred_struct_norm = tandem_model.predict_structure(input_color)
            pred_struct = structure_scaler.inverse_transform(pred_struct_norm.cpu().numpy())
            
            # Predict color from predicted structure (validation)
            pred_color_norm = tandem_model.validation_model(pred_struct_norm)
            pred_color = color_scaler.inverse_transform(pred_color_norm.cpu().numpy())
            
            results.append({
                'orig_x': orig_color[0],
                'orig_y': orig_color[1],
                'orig_Y': orig_color[2],
                'pred_x': pred_color[0, 0],
                'pred_y': pred_color[0, 1],
                'pred_Y': pred_color[0, 2],
                'orig_diameter': orig_struct[0],
                'orig_pitch': orig_struct[1],
                'orig_depth': orig_struct[2],
                'pred_diameter': pred_struct[0, 0],
                'pred_pitch': pred_struct[0, 1],
                'pred_depth': pred_struct[0, 2],
            })
    
    # Create evaluation DataFrame
    eval_df = pd.DataFrame(results)
    
    # Calculate Mean Absolute Errors for structures (main output)
    struct_mae = {
        'diameter_mae': np.abs(eval_df['orig_diameter'] - eval_df['pred_diameter']).mean(),
        'pitch_mae': np.abs(eval_df['orig_pitch'] - eval_df['pred_pitch']).mean(),
        'depth_mae': np.abs(eval_df['orig_depth'] - eval_df['pred_depth']).mean(),
    }
    
    # Calculate Mean Absolute Errors for colors (validation)
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
    print("Structure Prediction Metrics (Main Output):")
    print("  Mean Absolute Error (MAE):")
    for param, error in struct_mae.items():
        print(f"    {param}: {error:.2f}")
    print("  Root Mean Square Error (RMSE):")
    for param, error in struct_rmse.items():
        print(f"    {param}: {error:.2f}")
    print("  R² Score:")
    for param, r2 in struct_r2.items():
        print(f"    {param}: {r2:.4f}")
    
    print("\nColor Validation Metrics (Reconstruction Quality):")
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
    print(f"  Color Validation R²: {overall_color_r2:.4f}")
    print(f"  Combined R²: {(overall_struct_r2 + overall_color_r2) / 2:.4f}")
    
    # Plot evaluation results with R² values in titles
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Structure parameters (Main outputs)
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
    
    # Color parameters (Validation outputs)
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
        axes[1, i].set_xlabel(f'Original CIE {param}')
        axes[1, i].set_ylabel(f'Reconstructed CIE {param}')
        axes[1, i].set_title(f'CIE {param} Validation\nR² = {r2_val:.3f}, MAE = {mae_val:.4f}')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].legend()
    
    plt.tight_layout()
    plt.savefig('model_evaluation_color_to_structure.png', dpi=300, bbox_inches='tight')
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
    """Main training function - modified for color to structure mapping"""
    
    # Load and preprocess data (saves processed CSV automatically)
    colors, structures = load_and_preprocess_data('combined_nanohole_data70.csv', save_processed=True)
    
    # Create data visualization
    print("\nCreating data visualization...")
    df_processed = create_data_visualization(colors, structures, 'data_analysis_color_to_structure.png')
    
    # Train main model (color -> structure)
    main_model, color_scaler, structure_scaler, main_train_losses, main_val_losses = train_main_model(
        colors, structures, epochs=1000
    )

     # Step 2: Train inverse model (structure -> color)
    print("\nStep 2: Training Inverse Model...")
    inverse_model, _, _, inverse_train_losses, inverse_val_losses = train_inverse_model(
        colors, structures, epochs=1000
    )
    
    # Step 3: Train tandem model with both pre-trained models
    print("\nStep 3: Training Tandem Model...")
    tandem_model, tandem_train_losses, tandem_val_losses = train_tandem_model(
        colors, structures, main_model, inverse_model, 
        color_scaler, structure_scaler, epochs=1000  # Fewer epochs needed for fine-tuning
    )
    
    # Plot training history
    plot_training_history(
        (main_train_losses, main_val_losses),
        {'train': tandem_train_losses, 'val': tandem_val_losses},
        'training_history_color_to_structure.png'
    )
    
    # Evaluate model
    eval_results, metrics = evaluate_model(tandem_model, colors, structures, color_scaler, structure_scaler)
    
    # Save evaluation results
    eval_results.to_csv('model_evaluation_results_color_to_structure.csv', index=False)
    
    # Save comprehensive metrics to a separate file
    import json
    with open('model_metrics_color_to_structure.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary of key metrics
    print(f"\n" + "="*50)
    print("FINAL MODEL PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Overall Structure Prediction R² (Main Task): {metrics['overall_structure_r2']:.4f}")
    print(f"Overall Color Validation R² (Consistency): {metrics['overall_color_r2']:.4f}")
    print(f"Combined Model Performance R²: {metrics['overall_combined_r2']:.4f}")
    
    print("\nTraining completed successfully!")
    print("Saved files:")
    print("  - nanohole_processed_data_color_to_structure.csv (Main dataset: x, y, Y, diameter, pitch, depth)")
    print("  - nanohole_data_statistics_color_to_structure.csv (Summary statistics)")
    print("  - best_main_model_color_to_structure.pth (Trained main model)")
    print("  - best_tandem_model_color_to_structure.pth (Trained tandem model)")
    print("  - data_analysis_color_to_structure.png (Data visualization)")
    print("  - training_history_color_to_structure.png (Training curves)")
    print("  - model_evaluation_color_to_structure.png (Model performance)")
    print("  - model_evaluation_results_color_to_structure.csv (Detailed evaluation results)")
    print("  - model_metrics_color_to_structure.json (Comprehensive performance metrics)")
    
    return df_processed, tandem_model, color_scaler, structure_scaler, metrics

if __name__ == "__main__":
    main()